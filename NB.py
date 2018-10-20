import pandas as pd
import numpy as np
from ggplot import *
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# lead data to dataset var
dataset = pd.read_csv('weight-height.csv')

dataset['Gender'] = dataset['Gender'].replace('Male', 0)
dataset['Gender'] = dataset['Gender'].replace('Female',1)

X_train, X_test, y_train, y_test = train_test_split(dataset , dataset['Gender'], test_size=0.33, random_state=42)

mean_male = X_train[['Height', 'Weight']][X_train['Gender'] == 0].mean()
mean_female = X_train[['Height', 'Weight']][X_train['Gender'] == 1].mean()

var_male = X_train[['Height', 'Weight']][X_train['Gender'] == 0].var()
var_female = X_train[['Height', 'Weight']][X_train['Gender'] == 1].var()


# define likelihood


def likelihood(feature, mean, variance):
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp((-(feature - mean) ** 2) / (2 * variance))


print('Mean values for male features\n===\n{}'.format(mean_male))
print()
print('Mean values for female features\n===\n{}'.format(mean_female))
print()
print('Variance values for male features\n===\n{}'.format(var_male))
print()
print('Variance values for female features\n===\n{}'.format(var_female))

priors = np.array([1 / np.unique(X_train.Gender).shape[0]] * np.unique(X_train.Gender).shape[0])

predicted_class = []
examin_test = []

for num in range(len(X_test)):
    Likelihood_H = likelihood(feature=X_test.iloc[num]['Height'],
                 mean=np.array([mean_male['Height'], mean_female['Height']]),
                 variance=np.array([var_male['Height'], var_female['Height']]))
    Likelihood_W = likelihood(feature=X_test.iloc[num]['Weight'],
                 mean=np.array([mean_male['Weight'], mean_female['Weight']]),
                 variance=np.array([var_male['Weight'], var_female['Weight']]))
    prediction = priors * Likelihood_H * Likelihood_W
    predicted_class.append(np.argmax(prediction))
    if predicted_class[num] == X_test.iloc[num]['Gender']:
        examin_test.append(1)
    else:
        examin_test.append(0)


print('accuracry:',np.sum(examin_test)/len(X_test)*100)

plot = ggplot(dataset, aes(x='Weight',y ='Height', colour ='factor(Gender)', group = 'factor(Gender)')) + geom_density()
print(plot)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted_class)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes='factor(Gender)',
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes='factor(Gender)', normalize=True,
                      title='Normalized confusion matrix')

plt.show()