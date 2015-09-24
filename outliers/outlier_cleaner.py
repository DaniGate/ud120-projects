#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error)
    """

    cleaned_data = []

    ### your code goes here
    if len(predictions) != len(ages) or len(ages) != len(net_worths):
        print "List of values have different lengths!"
        return

    nAll = len(predictions)
    nRemove = int(0.1*nAll)
    error = []
    sorted_error = []

    for i in range(nAll):
        error.append(predictions[i][0] - net_worths[i][0])
        sorted_error.append(abs(error[i]))
        cleaned_data.append((ages[i][0], net_worths[i][0], error[i]))

    sorted_error.sort()
    print "Mean error: ",sum(sorted_error)/len(sorted_error)

    for i in range(nAll):
        try:
            error.index(sorted_error[i])
        except:
            ### Recover the signs
            sorted_error[i] *= -1.


    for i in range(nAll-1,nAll-nRemove-1,-1):
        del(cleaned_data[error.index(sorted_error[i])])
        error.remove(sorted_error[i]) ### Need to remove outliers also from error list to keep indexes consistent

    print nRemove," outliers successfully removed"
    # print "errors: ",error
    # print "sorted errors: ",sorted_error
    # print "Cleaned data: "
    # print cleaned_data

    return cleaned_data
