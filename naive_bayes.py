# Name: Rhea D'Souza
# Username: dsouzrhea

import pandas as pd
import sys

class NaiveBayesClassifier:
    def __init__(self):
        self.prob_table = None
        self.features = None

    def fit(self, training_set):
        self.features = training_set.drop(columns=['class']).columns
        self._train(training_set)
        return self

    def _initialise_unique_feature_class_combos(self, class_domain, training_set):
        # Log all unique combinations of features, feature_values and class_labels into dictionary and get domains
        # of all features
        feature_domains = {}
        count = {}
        for y in class_domain:
            count[y] = 1  # Initialize count numbers to 1

            for feature in self.features:
                feature_domain = set(training_set[feature])
                feature_domains[feature] = feature_domain

                for xi in feature_domain:
                    count[(feature, xi, y)] = 1
        return count, feature_domains

    def _populate_count_dict_with_train_data(self, count, training_set):
        # Count the number of each class and feature_value pairs based on the training instances
        for i in range(len(training_set)):
            instance = training_set.iloc[i]
            y = instance['class']
            count[y] += 1

            for feature in self.features:
                count[(feature, instance[feature], y)] += 1

        return count

    def _calculate_denominator_values(self, class_domain, feature_domains, count):
        # Calculate the total/denominators
        num_instances = 0
        total = {}
        for y in class_domain:
            num_instances += count[y]

            for feature in self.features:
                total[(feature, y)] = 0

                for xi in feature_domains[feature]:
                    total[(feature, y)] += count[(feature, xi, y)]
        return num_instances, total

    def _calculate_probabilities(self, class_domain, feature_domains, count, total, num_instances):
        # Calculate the probabilities from the counting numbers
        prob_table = {}
        for y in class_domain:
            prob_table[y] = count[y] / num_instances
            for feature in self.features:
                for xi in feature_domains[feature]:
                    prob_table[(feature, xi, y)] = count[(feature, xi, y)] / total[(feature, y)]

        self.prob_table = prob_table

    def _train(self, training_set):
        # Initialise feature domain info storage
        self.class_domain = set(training_set['class'])

        # Log all unique combinations of features, feature_values and class_labels into dictionary and get domains
        # of all features
        count, feature_domains = self._initialise_unique_feature_class_combos(self.class_domain, training_set)

        # Count the number of each class and feature_value pairs based on the training instances
        count = self._populate_count_dict_with_train_data(count, training_set)

        # Calculate the total/denominators
        num_instances, total = self._calculate_denominator_values(self.class_domain, feature_domains, count)

        # Calculate the probabilities from the counting numbers
        self._calculate_probabilities(self.class_domain, feature_domains, count, total, num_instances)

    def _calculate_class_score(self, test_instance, class_label):
        """
        Calculate the score of a class for a given test instance.
        :param test_instance: Test instance containing feature values.
        :param class_label: Class label for which the score is to be calculated.
        :return Score of the class for the given test instance.
        """
        score = self.prob_table[class_label]
        for feature in self.features:
            score *= self.prob_table.get((feature, test_instance[feature], class_label), 0)
        return score

    def predict(self, test_instance):
        """
        Predict the class of test instance
        :param test_instance:  Data containing features to classify based on
        :return: Classification predication of test instance
        """
        if self.prob_table is None:
            raise RuntimeError("Model not trained yet. Please train the model using the fit() method.")

        scores = {}
        for y in self.class_domain:
            scores[y] = self._calculate_class_score(test_instance, y)

        # Predict the class with the highest score
        predicted_class = max(scores, key=scores.get)
        return scores, predicted_class


def test_and_accuracy(train_or_test: str, nb_classifier: NaiveBayesClassifier, test_data: pd.DataFrame):
    """
    Runs the given pandas DataFrame instance by instance through the predict method for the Naive Bayes classifier.
    Prints both the prediction for each instance and the prediction accuracy of the entire given set.
    :param train_or_test: str indicating the dataset being run i.e. {train data, test data}
    :param nb_classifier: the pretrained classifier instance
    :param test_data: the dataset that will be used to test the classification model
    :return: None
    """

    print(f"\n\n\nPrinting {train_or_test} Data Results...")
    print("===========================================")

    # Classify test instances and print predictions
    accuracy = 0
    print("Predictions:")
    for _, test_instance in test_data.iterrows():
        class_score, prediction = nb_classifier.predict(test_instance)
        print(f"Instance No. {test_instance['Unnamed: 0']}\nActual Class: {test_instance['class']}\nPrediction: {prediction}\nScore: {class_score}\n")

        accuracy += prediction == test_instance['class']

    accuracy = accuracy / len(test_data)
    print(f"{train_or_test} accuracy is {accuracy * 100:.2f}%")


def print_feature_probabilities(prob_table: dict):
    """
    Prints the probability of all combinations of each feature and class.
    :param prob_table: Dictionary containing the probability of each feature and class.
    :return: None
    """
    print('~~~~~~~~~~~~~~~~~~~~Printing feature probabilities~~~~~~~~~~~~~~~~~~~~')
    for prob in prob_table.keys():
        if prob == 'recurrence-events' or prob == 'no-recurrence-events':
            print(f"P(class = {prob}): {prob_table[prob]:.4f}")
            continue
        print(f"P({prob[0]} = {prob[1]} | class = {prob[2]}): {prob_table[prob]:.4f}")
    print('~~~~~~~~~~~~~~~Finished printing feature probabilities~~~~~~~~~~~~~~~')


def main(training_file, test_file):
    # Load training data
    train_data = pd.read_csv('A3_part1data/' + training_file)
    training_data = train_data.drop(columns=['Unnamed: 0'])  # Remove the instance ID column from the training data
    # feature_values = {
    #     'age': ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
    #     'menopause': ['lt40', 'ge40', 'premeno'],
    #     'tumor-size': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
    #                    '55-59'],
    #     'inv-nodes': ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32',
    #                   '33-35', '36-39'],
    #     'node-caps': ['yes', 'no'],
    #     'deg-malig': ['1', '2', '3'],
    #     'breast': ['left', 'right'],
    #     'breast-quad': ['left up', 'left low', 'right up', 'right low', 'central'],
    #     'irradiat': ['yes', 'no']
    # }

    # Initialize and train the Naive Bayes classifier
    nb_classifier = NaiveBayesClassifier().fit(training_data)
    print_feature_probabilities(nb_classifier.prob_table)

    # Load test data
    test_data = pd.read_csv('A3_part1data/' + test_file)

    # Classify test instances and print predictions
    test_and_accuracy('Train', nb_classifier, train_data)
    test_and_accuracy('Test', nb_classifier, test_data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python naive_bayes.py <training_file> <test_file>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
