
🖥️  Device: cuda
   GPU: Tesla T4
   VRAM: 15.6 GB

############################################################
  TRAINING: efficientnet_b0
############################################################

  Class mapping: {'HDPE': 0, 'LDPE': 1, 'PET': 2, 'PP': 3, 'PS': 4, 'PVC': 5}
  Train: 1834 images, 58 batches
  Val:   226 images, 8 batches
  Class counts: {'HDPE': np.int64(403), 'LDPE': np.int64(412), 'PET': np.int64(324), 'PP': np.int64(119), 'PS': np.int64(396), 'PVC': np.int64(180)}
Downloading: "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth" to /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth
100%|███████████████████████████████████████| 20.5M/20.5M [00:00<00:00, 139MB/s]

============================================================
  Stage 1: Feature Extraction (efficientnet_b0)
============================================================
  Total params: 4,337,026
  Trainable:    329,478 (7.6%)

  Epoch  1/10 │ Train: 0.7939 loss=0.6255 │ Val: 0.8496 loss=0.4186 │ LR: 1.00e-03 │ 21.6s ✅ best
  Epoch  2/10 │ Train: 0.8899 loss=0.3502 │ Val: 0.8717 loss=0.3416 │ LR: 9.76e-04 │ 16.7s ✅ best
  Epoch  3/10 │ Train: 0.8839 loss=0.3542 │ Val: 0.8584 loss=0.3439 │ LR: 9.05e-04 │ 15.9s (1/10)
  Epoch  4/10 │ Train: 0.9117 loss=0.2704 │ Val: 0.8894 loss=0.2978 │ LR: 7.94e-04 │ 15.4s ✅ best
  Epoch  5/10 │ Train: 0.9182 loss=0.2406 │ Val: 0.8938 loss=0.3286 │ LR: 6.55e-04 │ 15.3s ✅ best
  Epoch  6/10 │ Train: 0.9177 loss=0.2394 │ Val: 0.8805 loss=0.3213 │ LR: 5.00e-04 │ 15.2s (1/10)
  Epoch  7/10 │ Train: 0.9166 loss=0.2573 │ Val: 0.8850 loss=0.2944 │ LR: 3.45e-04 │ 14.8s (2/10)
  Epoch  8/10 │ Train: 0.9286 loss=0.2209 │ Val: 0.8850 loss=0.2885 │ LR: 2.06e-04 │ 15.0s (3/10)
  Epoch  9/10 │ Train: 0.9378 loss=0.2048 │ Val: 0.8850 loss=0.2908 │ LR: 9.55e-05 │ 14.7s (4/10)
  Epoch 10/10 │ Train: 0.9308 loss=0.2128 │ Val: 0.8850 loss=0.2899 │ LR: 2.45e-05 │ 15.1s (5/10)

  Best val accuracy: 0.8938

============================================================
  Stage 2: Fine-Tuning (efficientnet_b0)
============================================================
  Total params: 4,337,026
  Trainable:    3,485,218 (80.4%)

  Epoch  1/20 │ Train: 0.8795 loss=0.3758 │ Val: 0.8761 loss=0.3429 │ LR: 1.00e-04 │ 15.2s ✅ best
  Epoch  2/20 │ Train: 0.9160 loss=0.2573 │ Val: 0.8761 loss=0.3441 │ LR: 9.05e-05 │ 15.4s (1/7)
  Epoch  3/20 │ Train: 0.9242 loss=0.2323 │ Val: 0.8850 loss=0.3029 │ LR: 6.55e-05 │ 15.1s ✅ best
  Epoch  4/20 │ Train: 0.9537 loss=0.1571 │ Val: 0.8805 loss=0.3730 │ LR: 3.45e-05 │ 15.0s (1/7)
  Epoch  5/20 │ Train: 0.9466 loss=0.1641 │ Val: 0.8850 loss=0.3237 │ LR: 9.55e-06 │ 14.9s (2/7)
  Epoch  6/20 │ Train: 0.9368 loss=0.1698 │ Val: 0.8850 loss=0.3707 │ LR: 1.00e-04 │ 15.4s (3/7)
  Epoch  7/20 │ Train: 0.9558 loss=0.1411 │ Val: 0.9115 loss=0.3259 │ LR: 9.05e-05 │ 15.1s ✅ best
  Epoch  8/20 │ Train: 0.9602 loss=0.1204 │ Val: 0.8894 loss=0.3420 │ LR: 6.55e-05 │ 15.0s (1/7)
  Epoch  9/20 │ Train: 0.9689 loss=0.1030 │ Val: 0.8761 loss=0.3328 │ LR: 3.45e-05 │ 14.9s (2/7)
  Epoch 10/20 │ Train: 0.9711 loss=0.0813 │ Val: 0.8761 loss=0.3500 │ LR: 9.55e-06 │ 14.9s (3/7)
  Epoch 11/20 │ Train: 0.9700 loss=0.0789 │ Val: 0.8982 loss=0.3892 │ LR: 1.00e-04 │ 14.8s (4/7)
  Epoch 12/20 │ Train: 0.9651 loss=0.1003 │ Val: 0.9115 loss=0.3664 │ LR: 9.05e-05 │ 15.5s (5/7)
  Epoch 13/20 │ Train: 0.9826 loss=0.0602 │ Val: 0.8982 loss=0.4505 │ LR: 6.55e-05 │ 15.4s (6/7)
  Epoch 14/20 │ Train: 0.9787 loss=0.0638 │ Val: 0.8982 loss=0.4548 │ LR: 3.45e-05 │ 15.3s (7/7)

  ⏹ Early stopping at epoch 14 (no improvement for 7 epochs)

  Best val accuracy: 0.9115
  📈 Saved: /kaggle/working/outputs/training_curves_efficientnet_b0.png
  💾 History saved: /kaggle/working/outputs/history_efficientnet_b0.json
  💾 Best model saved: /kaggle/working/models/best_efficientnet_b0.pth

############################################################
  TRAINING: resnet50
############################################################

  Class mapping: {'HDPE': 0, 'LDPE': 1, 'PET': 2, 'PP': 3, 'PS': 4, 'PVC': 5}
  Train: 1834 images, 58 batches
  Val:   226 images, 8 batches
  Class counts: {'HDPE': np.int64(403), 'LDPE': np.int64(412), 'PET': np.int64(324), 'PP': np.int64(119), 'PS': np.int64(396), 'PVC': np.int64(180)}
Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
100%|███████████████████████████████████████| 97.8M/97.8M [00:00<00:00, 204MB/s]

============================================================
  Stage 1: Feature Extraction (resnet50)
============================================================
  Total params: 24,034,118
  Trainable:    526,086 (2.2%)

  Epoch  1/10 │ Train: 0.7246 loss=0.7624 │ Val: 0.8142 loss=0.5309 │ LR: 1.00e-03 │ 15.9s ✅ best
  Epoch  2/10 │ Train: 0.8348 loss=0.4703 │ Val: 0.8363 loss=0.5721 │ LR: 9.76e-04 │ 16.0s ✅ best
  Epoch  3/10 │ Train: 0.8332 loss=0.4867 │ Val: 0.8451 loss=0.4632 │ LR: 9.05e-04 │ 15.5s ✅ best
  Epoch  4/10 │ Train: 0.8675 loss=0.3961 │ Val: 0.8540 loss=0.4298 │ LR: 7.94e-04 │ 16.3s ✅ best
  Epoch  5/10 │ Train: 0.8642 loss=0.4048 │ Val: 0.8805 loss=0.3418 │ LR: 6.55e-04 │ 15.9s ✅ best
  Epoch  6/10 │ Train: 0.8795 loss=0.3512 │ Val: 0.8584 loss=0.3532 │ LR: 5.00e-04 │ 16.2s (1/10)
  Epoch  7/10 │ Train: 0.8664 loss=0.3671 │ Val: 0.8584 loss=0.4047 │ LR: 3.45e-04 │ 15.6s (2/10)
  Epoch  8/10 │ Train: 0.8822 loss=0.3400 │ Val: 0.8673 loss=0.3760 │ LR: 2.06e-04 │ 15.8s (3/10)
  Epoch  9/10 │ Train: 0.8860 loss=0.3324 │ Val: 0.8628 loss=0.3700 │ LR: 9.55e-05 │ 15.6s (4/10)
  Epoch 10/10 │ Train: 0.8959 loss=0.3234 │ Val: 0.8673 loss=0.3684 │ LR: 2.45e-05 │ 15.9s (5/10)

  Best val accuracy: 0.8805

============================================================
  Stage 2: Fine-Tuning (resnet50)
============================================================
  Total params: 24,034,118
  Trainable:    15,490,822 (64.5%)

  Epoch  1/20 │ Train: 0.8762 loss=0.3835 │ Val: 0.8761 loss=0.4423 │ LR: 1.00e-04 │ 16.5s ✅ best
  Epoch  2/20 │ Train: 0.9166 loss=0.2799 │ Val: 0.8982 loss=0.3332 │ LR: 9.05e-05 │ 16.0s ✅ best
  Epoch  3/20 │ Train: 0.9406 loss=0.1953 │ Val: 0.8938 loss=0.3122 │ LR: 6.55e-05 │ 16.5s (1/7)
  Epoch  4/20 │ Train: 0.9520 loss=0.1390 │ Val: 0.9027 loss=0.3359 │ LR: 3.45e-05 │ 16.2s ✅ best
  Epoch  5/20 │ Train: 0.9733 loss=0.0886 │ Val: 0.9071 loss=0.3152 │ LR: 9.55e-06 │ 16.2s ✅ best
  Epoch  6/20 │ Train: 0.9558 loss=0.1429 │ Val: 0.8982 loss=0.4808 │ LR: 1.00e-04 │ 16.2s (1/7)
  Epoch  7/20 │ Train: 0.9482 loss=0.1725 │ Val: 0.8894 loss=0.4954 │ LR: 9.05e-05 │ 16.5s (2/7)
  Epoch  8/20 │ Train: 0.9722 loss=0.0928 │ Val: 0.8982 loss=0.5653 │ LR: 6.55e-05 │ 16.1s (3/7)
  Epoch  9/20 │ Train: 0.9766 loss=0.0735 │ Val: 0.9027 loss=0.5514 │ LR: 3.45e-05 │ 16.2s (4/7)
  Epoch 10/20 │ Train: 0.9847 loss=0.0381 │ Val: 0.8982 loss=0.5253 │ LR: 9.55e-06 │ 16.6s (5/7)
  Epoch 11/20 │ Train: 0.9755 loss=0.0884 │ Val: 0.9027 loss=0.6457 │ LR: 1.00e-04 │ 16.4s (6/7)
  Epoch 12/20 │ Train: 0.9695 loss=0.1089 │ Val: 0.9115 loss=0.5195 │ LR: 9.05e-05 │ 16.5s ✅ best
  Epoch 13/20 │ Train: 0.9831 loss=0.0640 │ Val: 0.9027 loss=0.6222 │ LR: 6.55e-05 │ 16.7s (1/7)
  Epoch 14/20 │ Train: 0.9880 loss=0.0401 │ Val: 0.9027 loss=0.6508 │ LR: 3.45e-05 │ 16.4s (2/7)
  Epoch 15/20 │ Train: 0.9935 loss=0.0233 │ Val: 0.8982 loss=0.6516 │ LR: 9.55e-06 │ 16.8s (3/7)
  Epoch 16/20 │ Train: 0.9902 loss=0.0412 │ Val: 0.9027 loss=0.6161 │ LR: 1.00e-04 │ 17.0s (4/7)
  Epoch 17/20 │ Train: 0.9880 loss=0.0460 │ Val: 0.8451 loss=1.1323 │ LR: 9.05e-05 │ 16.3s (5/7)
  Epoch 18/20 │ Train: 0.9913 loss=0.0433 │ Val: 0.8982 loss=0.5419 │ LR: 6.55e-05 │ 16.4s (6/7)
  Epoch 19/20 │ Train: 0.9896 loss=0.0343 │ Val: 0.9159 loss=0.6751 │ LR: 3.45e-05 │ 16.6s ✅ best
  Epoch 20/20 │ Train: 0.9918 loss=0.0236 │ Val: 0.8938 loss=0.5542 │ LR: 9.55e-06 │ 16.3s (1/7)

  Best val accuracy: 0.9159
  📈 Saved: /kaggle/working/outputs/training_curves_resnet50.png
  💾 History saved: /kaggle/working/outputs/history_resnet50.json
  💾 Best model saved: /kaggle/working/models/best_resnet50.pth

============================================================
  TRAINING COMPLETE — 14.4 minutes total
============================================================

  ┌─────────────────┬───────────────┐
  │     Model       │ Best Val Acc  │
  ├─────────────────┼───────────────┤
  │ efficientnet_b0 │     0.9115   │
  │ resnet50        │     0.9159   │ 🏆
  └─────────────────┴───────────────┘

  Models saved to: /kaggle/working/models/
  Plots saved to:  /kaggle/working/outputs/

---


🖥️  Device: cuda
  Test set: 235 images, classes: ['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'PVC']
  ✅ Loaded: /kaggle/working/models/best_efficientnet_b0.pth

============================================================
  EVALUATING: efficientnet_b0
============================================================

  ┌────────────────────┬──────────┐
  │       Metric       │  Value   │
  ├────────────────────┼──────────┤
  │ Accuracy           │  0.8894  │
  │ Macro F1           │  0.8641  │
  │ Weighted F1        │  0.8909  │
  │ Cohen's Kappa      │  0.8636  │
  │ Top-3 Accuracy     │  0.9745  │
  └────────────────────┴──────────┘

  Classification Report:
              precision    recall  f1-score   support

        HDPE     0.8868    0.9216    0.9038        51
        LDPE     0.9623    0.9623    0.9623        53
         PET     0.7872    0.8810    0.8315        42
          PP     0.5882    0.6250    0.6061        16
          PS     0.9767    0.8400    0.9032        50
         PVC     1.0000    0.9565    0.9778        23

    accuracy                         0.8894       235
   macro avg     0.8669    0.8644    0.8641       235
weighted avg     0.8959    0.8894    0.8909       235

  📄 Saved: /kaggle/working/outputs/classification_report_efficientnet_b0.txt
  📊 Saved: /kaggle/working/outputs/confusion_matrix_efficientnet_b0.png
  📊 Saved: /kaggle/working/outputs/per_class_f1_efficientnet_b0.png
  📊 Saved: /kaggle/working/outputs/confidence_dist_efficientnet_b0.png
  📊 Saved: /kaggle/working/outputs/misclassified_efficientnet_b0.png
  ✅ Loaded: /kaggle/working/models/best_resnet50.pth

============================================================
  EVALUATING: resnet50
============================================================

  ┌────────────────────┬──────────┐
  │       Metric       │  Value   │
  ├────────────────────┼──────────┤
  │ Accuracy           │  0.8723  │
  │ Macro F1           │  0.8278  │
  │ Weighted F1        │  0.8703  │
  │ Cohen's Kappa      │  0.8422  │
  │ Top-3 Accuracy     │  0.9787  │
  └────────────────────┴──────────┘

  Classification Report:
              precision    recall  f1-score   support

        HDPE     0.9783    0.8824    0.9278        51
        LDPE     0.8929    0.9434    0.9174        53
         PET     0.7451    0.9048    0.8172        42
          PP     0.5000    0.3750    0.4286        16
          PS     0.9167    0.8800    0.8980        50
         PVC     1.0000    0.9565    0.9778        23

    accuracy                         0.8723       235
   macro avg     0.8388    0.8237    0.8278       235
weighted avg     0.8738    0.8723    0.8703       235

  📄 Saved: /kaggle/working/outputs/classification_report_resnet50.txt
  📊 Saved: /kaggle/working/outputs/confusion_matrix_resnet50.png
  📊 Saved: /kaggle/working/outputs/per_class_f1_resnet50.png
  📊 Saved: /kaggle/working/outputs/confidence_dist_resnet50.png
  📊 Saved: /kaggle/working/outputs/misclassified_resnet50.png

============================================================
  MODEL COMPARISON
============================================================

  ┌────────────────────┬─────────────────┬─────────────────┐
  │       Metric       │ EfficientNet-B0 │    ResNet50     │
  ├────────────────────┼─────────────────┼─────────────────┤
  │ Accuracy           │  0.8894 🏆      │  0.8723         │
  │ Macro F1           │  0.8641 🏆      │  0.8278         │
  │ Weighted F1        │  0.8909 🏆      │  0.8703         │
  │ Cohen's Kappa      │  0.8636 🏆      │  0.8422         │
  │ Top-3 Accuracy     │  0.9745         │  0.9787 🏆      │
  └────────────────────┴─────────────────┴─────────────────┘

  🏆 Best model: efficientnet_b0 (Macro F1 = 0.8641)
  💾 Saved: /kaggle/working/outputs/evaluation_results.json