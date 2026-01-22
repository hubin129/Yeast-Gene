H1p=read.table("/home/lizhu/model/ATAC&RNA/fasta/pos_fa/H1_5mer_pcounts.txt", header=T, sep="\t")
H2p=read.table("/home/lizhu/model/ATAC&RNA/fasta/pos_fa/H2_5mer_pcounts.txt", header=T, sep="\t")
H3p=read.table("/home/lizhu/model/ATAC&RNA/fasta/pos_fa/H3_5mer_pcounts.txt", header=T, sep="\t")
H4p=read.table("/home/lizhu/model/ATAC&RNA/fasta/pos_fa/H4_5mer_pcounts.txt", header=T, sep="\t")
H5p=read.table("/home/lizhu/model/ATAC&RNA/fasta/pos_fa/H5_5mer_pcounts.txt", header=T, sep="\t")
H6p=read.table("/home/lizhu/model/ATAC&RNA/fasta/pos_fa/H6_5mer_pcounts.txt", header=T, sep="\t")
H1n=read.table("/home/lizhu/model/ATAC&RNA/fasta/neg_fa/H1_5mer_ncounts.txt", header=T, sep="\t")
H2n=read.table("/home/lizhu/model/ATAC&RNA/fasta/neg_fa/H2_5mer_ncounts.txt", header=T, sep="\t")
H3n=read.table("/home/lizhu/model/ATAC&RNA/fasta/neg_fa/H3_5mer_ncounts.txt", header=T, sep="\t")
H4n=read.table("/home/lizhu/model/ATAC&RNA/fasta/neg_fa/H4_5mer_ncounts.txt", header=T, sep="\t")
H5n=read.table("/home/lizhu/model/ATAC&RNA/fasta/neg_fa/H5_5mer_ncounts.txt", header=T, sep="\t")
H6n=read.table("/home/lizhu/model/ATAC&RNA/fasta/neg_fa/H6_5mer_ncounts.txt", header=T, sep="\t")
data_p=rbind(H1p, H2p, H3p, H4p, H5p, H6p)
data_n=rbind(H1n, H2n, H3n, H4n, H5n, H6n)
data_p$type <- 1  
data_n$type <- 0

set.seed(111)

data_n <- sample_n(data_n, size = 1200)
data=rbind(data_p, data_n)
set.seed(111)
train_index=createDataPartition(data$type, p = 0.9, list = FALSE)

train_data=data[train_index, ]
test_data=data[-train_index, ]
library(caret)
library(dplyr)
library(e1071)
library(randomForest)
library(gbm)
library(ROSE)
library(ggplot2)
set.seed(111)
# 转换类型为因子并确保因子水平一致
train_data$type <- as.factor(train_data$type)
test_data$type <- as.factor(test_data$type)
levels(train_data$type) <- make.names(levels(factor(c(0, 1))))
levels(test_data$type) <- levels(train_data$type)
# 训练控制设置
control <- trainControl(method = "cv", number = 10, savePredictions = "final", classProbs = TRUE)

# 训练模型
model <- train(type ~ ., data = train_data, method = "rf", trControl = control, ntree = 5000, importance = TRUE)
# 进行模型预测和评估
predictions = predict(model, test_data)
performance = confusionMatrix(predictions, test_data$type)
# 输出模型的预测性能
print(performance)
# 结果可视化：ROC曲线和AUC
probabilities <- predict(model, test_data, type = "prob")[,2]  # 获取分类概率
roc_curve <- roc(response = test_data$type, probabilities)
plot(roc_curve, main="ROC Curve")
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))
#####################################################SVM
svm_model <- train(type ~ ., data = train_data, method = "svmRadial", trControl = control, preProcess = "scale", tuneLength = 5)
svm_predictions <- predict(svm_model, test_data)
svm_cm <- confusionMatrix(svm_predictions, test_data$type)
print(svm_cm)
# 结果可视化：ROC曲线和AUC
probabilities_svm <- predict(svm_model, test_data, type = "prob")[,2]  # 获取分类概率
roc.svm_curve <- roc(response = test_data$type, probabilities_svm)
plot(roc_curve, main="svm_ROC Curve")
auc.svm_value <- auc(roc.svm_curve)
print(paste("AUC:", auc.svm_value))
#####################################################GBM
gbm_model <- train(type ~ ., data = train_data, method = "gbm", trControl = control, verbose = FALSE)
gbm_predictions <- predict(gbm_model, test_data)
gbm_cm <- confusionMatrix(gbm_predictions, test_data$type)
print(gbm_cm)
# 结果可视化：ROC曲线和AUC
probabilities_gbm <- predict(gbm_model, test_data, type = "prob")[,2]  # 获取分类概率
roc.gbm_curve <- roc(response = test_data$type, probabilities_gbm)
pdf("/home/lizhu/model/ATAC&RNA/fig/gbm_ROC",width = 10, height = 6)
plot(roc.gbm_curve, main="GBM_ROC Curve")
auc.gbm_value <- auc(roc.gbm_curve)
print(paste("AUC:", auc.gbm_value))
text(x = 0.6, y = 0.2, labels = paste("AUC:", round(auc.gbm_value, 3)), cex = 1.2, col = "black")
dev.off()
# 生成混淆矩阵
confusion_matrix <- confusionMatrix(gbm_predictions, test_data$type)
# 绘制混淆矩阵
pdf("/home/lizhu/model/ATAC&RNA/fig/gbm_Matrix",width = 10, height = 6)
confusion_df <- as.data.frame(confusion_matrix$table)
colnames(confusion_df) <- c("Reference", "Prediction", "Frequency")
ggplot(confusion_df, aes(x=Reference, y=Frequency, fill=Prediction)) +
  geom_bar(stat="identity", position=position_dodge()) +
  theme_minimal() +
  labs(title="GBM_Confusion Matrix", x="Actual Class", y="Count", fill="Predicted Class")
dev.off()
# 计算评价指标
accuracy <- confusion_matrix$overall['Accuracy']
precision <- confusion_matrix$byClass['Precision']
recall <- confusion_matrix$byClass['Recall']
f1_score <- 2 * (precision * recall) / (precision + recall)  # 计算F1分数
# 绘制评价指标图
pdf("/home/lizhu/model/ATAC&RNA/fig/gbm_assess",width = 10, height = 6)
metrics_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
  Value = c(accuracy, precision, recall, f1_score)
)
ggplot(metrics_df, aes(x=Metric, y=Value, fill=Metric)) +
  geom_bar(stat="identity", position=position_dodge(), color="black") +
  ylim(0, 1) +
  theme_minimal() +
  labs(title="Model Performance Metrics", x="Metric", y="Value") +
  geom_text(aes(label=round(Value, 2)), vjust=-0.3, color="black")
dev.off()
##################
library(nnet)
library(e1071)
library(class)
library(kernlab)
library(adabag)
library(dplyr)
#############################LR
logistic_model <- train(type ~ ., data = train_data, method = "glm", family = "binomial", trControl = control)
logistic_predictions <- predict(logistic_model, test_data)
logistic_cm <- confusionMatrix(logistic_predictions, test_data$type)
print(logistic_cm)
# 结果可视化：ROC曲线和AUC
probabilities_LR <- predict(logistic_model, test_data, type = "prob")[,2]  # 获取分类概率
roc.LR_curve <- roc(response = test_data$type, probabilities_LR)
pdf("/home/lizhu/model/ATAC&RNA/fig/LR_ROC",width = 10, height = 6)
plot(roc.LR_curve, main="LR_ROC Curve")
auc.LR_value <- auc(roc.LR_curve)
print(paste("AUC:", auc.LR_value))
text(x = 0.6, y = 0.2, labels = paste("AUC:", round(auc.LR_value, 3)), cex = 1.2, col = "black")

dev.off()
###############################CART
tree_model <- train(type ~ ., data = train_data, method = "rpart", trControl = control)
tree_predictions <- predict(tree_model, test_data)
tree_cm <- confusionMatrix(tree_predictions, test_data$type)
print(tree_cm)
# 结果可视化：ROC曲线和AUC
probabilities_rpart <- predict(tree_model, test_data, type = "prob")[,2]  # 获取分类概率
roc.rpart_curve <- roc(response = test_data$type, probabilities_rpart)
pdf("/home/lizhu/model/ATAC&RNA/fig/rpart_ROC",width = 10, height = 6)
plot(roc.rpart_curve, main="tree_ROC Curve")
auc.rpart_value <- auc(roc.rpart_curve)
print(paste("AUC:", auc.rpart_value))
text(x = 0.6, y = 0.2, labels = paste("AUC:", round(auc.rpart_value, 3)), cex = 1.2, col = "black")
dev.off()
############################KNN
knn_model <- train(type ~ ., data = train_data, method = "knn", trControl = control)
knn_predictions <- predict(knn_model, test_data)
knn_cm <- confusionMatrix(knn_predictions, test_data$type)
print(knn_cm)
# 结果可视化：ROC曲线和AUC
probabilities_knn <- predict(knn_model, test_data, type = "prob")[,2]  # 获取分类概率
roc.knn_curve <- roc(response = test_data$type, probabilities_knn)
pdf("/home/lizhu/model/ATAC&RNA/fig/knn_ROC",width = 10, height = 6)
plot(roc.knn_curve, main="knn_ROC Curve")
auc.knn_value <- auc(roc.knn_curve)
print(paste("AUC:", auc.knn_value))
text(x = 0.6, y = 0.2, labels = paste("AUC:", round(auc.knn_value, 3)), cex = 1.2, col = "black")
dev.off()
#############################bayes
nb_model <- train(type ~ ., data = train_data, method = "naive_bayes", trControl = control)
nb_predictions <- predict(nb_model, test_data)
nb_cm <- confusionMatrix(nb_predictions, test_data$type)
print(nb_cm)
# 结果可视化：ROC曲线和AUC
probabilities_nb <- predict(nb_model, test_data, type = "prob")[,2]  # 获取分类概率
roc.nb_curve <- roc(response = test_data$type, probabilities_nb)
pdf("/home/lizhu/model/ATAC&RNA/fig/nb_ROC",width = 10, height = 6)
plot(roc.nb_curve, main="nb_ROC Curve")
auc.nb_value <- auc(roc.nb_curve)
print(paste("AUC:", auc.nb_value))
text(x = 0.6, y = 0.2, labels = paste("AUC:", round(auc.nb_value, 3)), cex = 1.2, col = "black")
dev.off()
###########################adaboost
ada_model <- train(type ~ ., data = train_data, method = "adaboost", trControl = control)
ada_predictions <- predict(ada_model, test_data)
ada_cm <- confusionMatrix(ada_predictions, test_data$type)
print(ada_cm)
# 结果可视化：ROC曲线和AUC
probabilities_ada <- predict(ada_model, test_data, type = "prob")[,2]  # 获取分类概率
roc.ada_curve <- roc(response = test_data$type, probabilities_ada)
pdf("/home/lizhu/model/ATAC&RNA/fig/ada_ROC",width = 10, height = 6)
plot(roc.nb_curve, main="ada_ROC Curve")
auc.ada_value <- auc(roc.ada_curve)
print(paste("AUC:", auc.ada_value))
text(x = 0.6, y = 0.2, labels = paste("AUC:", round(auc.ada_value, 3)), cex = 1.2, col = "black")
dev.off()
##################绘图
library(ggplot2)
library(pROC)
library(gridExtra)
# 生成ROC曲线数据
roc_data <- function(model, test_data, model_name, color) {
  predictions <- predict(model, test_data, type = "prob")[, 2]
  roc_curve <- roc(test_data$type, predictions)
  data.frame(t = roc_curve$thresholds, sens = roc_curve$sensitivities, spec = roc_curve$specificities, model = model_name, color = color)
}

# 数据准备
data_rf <- roc_data(model, test_data, "Random Forest", "red")
data_svm <- roc_data(svm_model, test_data, "SVM", "blue")
data_gbm <- roc_data(gbm_model, test_data, "GBM", "green")
data_lr <- roc_data(logistic_model, test_data, "Logistic Regression", "purple")
data_rpart <- roc_data(tree_model, test_data, "Decision Tree", "orange")
data_knn <- roc_data(knn_model, test_data, "KNN", "pink")
data_nb <- roc_data(nb_model, test_data, "Naive Bayes", "cyan")
data_ada <- roc_data(ada_model, test_data, "AdaBoost", "gray")

# 合并数据
all_data <- rbind(data_rf, data_svm, data_gbm, data_lr, data_rpart, data_knn, data_nb, data_ada)

# 设置PDF输出
pdf("/home/lizhu/model/ATAC&RNA/fig/all_models_ROC.pdf", width = 12, height = 10)

# 使用ggplot2绘制
p <- ggplot(all_data, aes(x = 1 - spec, y = sens, color = model, group = model)) +
  geom_line() +
  scale_color_manual(values = setNames(c("red", "blue", "green", "purple", "orange", "pink", "cyan", "gray"), 
                                       c("Random Forest", "SVM", "GBM", "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "AdaBoost"))) +
  labs(title = "ROC Curves for Different Models", x = "1 - Specificity", y = "Sensitivity") +
  theme_minimal()

print(p)

dev.off()
