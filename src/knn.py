import pandas


# k_value describes k in k-nn
# Currently columns_to_use expected to be only numeric columns, others will be added later
def measure_performance(train_set: pandas.DataFrame, test_set: pandas.DataFrame, k_value: int, columns_to_use: list):

    # Calculate similarity between train and test data
    # If its one of k neighbors, add it's productivity to
    # current_neighbors map

    for test_index, test_row in test_set.iterrows():

        current_similarities = dict()
        predicted_productivity = 0

        for train_index, train_row in train_set.iterrows():

            # TODO: Implement similarity calculation
            similarity = 0
            for column in columns_to_use:
                if test_row[column] == train_row[column]:
                    similarity += 1

            current_similarities[similarity] = train_row

        i = 1
        for key in sorted(current_similarities.keys(), reverse=True):
            predicted_productivity += float(current_similarities.get(key)['productivity'])
            i += 1
            if i == k_value:
                break

        predicted_productivity = predicted_productivity / float(k_value)
        print(predicted_productivity)

