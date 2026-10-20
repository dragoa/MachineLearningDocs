### Medical Diagnosis

**Priority:** Don't miss cases (minimize false negatives)

```
Choose: RECALL
Reason: Missing a patient = death (very costly)
Accept: Some false alarms (unnecessary tests are acceptable)
```
### Email Spam Filter

**Priority:** Don't delete legitimate emails (minimize false positives)

```
Choose: PRECISION
Reason: Deleting real email = user loses important message
Accept: Some spam slips through (annoying but acceptable)
```

### Fraud Detection

**Priority:** Catch fraud AND avoid false alarms

```
Choose: F1 SCORE (or balance both metrics)
Reason: Dataset heavily imbalanced (1% fraud, 99% legitimate)
Note: Accuracy would be useless (99% baseline by always predicting "legitimate")
```

### Image Classification (10 classes)

**Priority:** Choose averaging based on whether classes should count equally or by frequency

```
Choose: Macro F1 if every class matters equally; weighted F1 if class frequency should determine each class's contribution
Reason: The averaging choice determines how much rare classes affect the score
Note: Weighted F1 does not give rare classes the same influence as common classes
```