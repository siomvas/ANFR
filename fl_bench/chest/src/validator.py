import torch
from monai.inferers.inferer import SimpleInferer
from monai.metrics import ROCAUCMetric, compute_confusion_matrix_metric, get_confusion_matrix
from monai.transforms import Activations
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix, roc_curve
import numpy as np

# TODO: Add actual class names to the labels


class Validator():
    def __init__(self):
        self.inferer = SimpleInferer()
        self.transform_post = Activations(sigmoid=True)
        self.metric = ROCAUCMetric(average="none")

    def run(self, model, data_loader, logger=None, step=None):
        model.eval()
        y = []
        y_pred = []
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    batch["image"] = batch["image"].to("cuda:0")
                    batch["label"] = batch["label"].to("cuda:0")
                    batch["preds"] = self.inferer(batch["image"], model)
                    if self.transform_post is not None:
                        batch['preds'] = self.transform_post(batch['preds'])
                    y.append(batch["label"])
                    y_pred.append(batch["preds"])

                y = torch.cat(y)
                y_pred = torch.cat(y_pred)
                self.metric(batch["preds"], batch["label"])

                if logger is not None:
                    for idx in range(y_pred.shape[1]):
                        logger.add_pr_curve(
                            f"pr_curve_{idx}",
                            labels=y[:, idx],
                            predictions=y_pred[:, idx],
                            global_step=step
                        )

        raw_metrics = self.metric.aggregate()
        self.metric.reset()

        mean = 0.0
        metrics = {}
        for idx in range(len(raw_metrics)):
            mean += raw_metrics[idx]
            metrics["val_auroc_" + str(idx)] = raw_metrics[idx]
        metric_score = mean / len(raw_metrics)

        metrics["val_mean_auroc"] = metric_score

        # Youden's J statistic
        # to find the optimal threshold for each class
        y_pred_np = y_pred.numpy()
        y_np = y.numpy()
        optimal_thresholds = np.zeros(y.shape[1])
        for class_idx in range(y_np.shape[1]):
            fpr, tpr, thresholds = roc_curve(y.numpy(force=True)[:, class_idx], y_pred.numpy(force=True)[: class_idx])
            optimal_thresholds[class_idx] = thresholds[np.argmax(tpr - fpr)]
        metrics["optimal_thresholds"] = optimal_thresholds
        binarized_labels = (y_pred_np > optimal_thresholds).astype(np.int32)

        multi_label_confusion_matrix = get_confusion_matrix(y=y, y_pred=torch.tensor(binarized_labels))
        macro_f1 = 0.0
        macro_sens = 0.0
        macro_spec = 0.0
        macro_prec = 0.0
        macro_rec = 0.0
        metrics = ["f1", "specificity", "sensitivity", "precision", "recall"]
        for class_idx in range(multilabel_confusion_matrix.shape[1]):
            class_metrics = compute_confusion_matrix_metric(
                metrics, multi_label_confusion_matrix[:, class_idx, :, :]
            )
            macro_f1 += class_metrics[0]
            macro_sens += class_metrics[1]
            macro_spec += class_metrics[2]
            macro_prec += class_metrics[3]
            macro_rec += class_metrics[4]

        metrics["val_macro_f1"] = macro_f1 / y.shape[1]
        metrics["val_macro_sens"] = macro_sens / y.shape[1]
        metrics["val_macro_spec"] = macro_spec / y.shape[1]
        metrics["val_macro_prec"] = macro_prec / y.shape[1]
        metrics["val_macro_rec"] = macro_rec / y.shape[1]

        # Metrics calculation (macro) over the whole set
        confusion = multilabel_confusion_matrix(y_true=y_np, y_pred=binarized_labels)
        epsilon = 1e-7
        F1_disease = []
        accuracy_disease = []
        specificity_disease = []
        sensitivity_disease = []
        precision_disease = []
        total_f1_score = []
        total_accuracy = []
        total_specificity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        for idx, disease in enumerate(confusion):
            TN = disease[0, 0]
            FP = disease[0, 1]
            FN = disease[1, 0]
            TP = disease[1, 1]
            F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
            accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
            specificity_disease.append(TN / (TN + FP + epsilon))
            sensitivity_disease.append(TP / (TP + FN + epsilon))
            precision_disease.append(TP / (TP + FP + epsilon))

        # Macro averaging
        total_f1_score.append(np.stack(F1_disease))
        total_accuracy.append(np.stack(accuracy_disease))
        total_specificity_score.append(np.stack(specificity_disease))
        total_sensitivity_score.append(np.stack(sensitivity_disease))
        total_precision_score.append(np.stack(precision_disease))

        average_f1_score = np.stack(total_f1_score).mean(0)
        average_accuracy = np.stack(total_accuracy).mean(0)
        average_specificity = np.stack(total_specificity_score).mean(0)
        average_sensitivity = np.stack(total_sensitivity_score).mean(0)
        average_precision = np.stack(total_precision_score).mean(0)

        metrics["average_f1_score"] = average_f1_score
        metrics["average_accuracy"] = average_accuracy
        metrics["average_specificity"] = average_specificity
        metrics["average_sensitivity"] = average_sensitivity
        metrics["average_precision"] = average_precision

        return metrics
