#!/usr/bin/python

"""
    starter code for exploring the Enron dataset (emails + finances)
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle
import math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Number of people in the E+F dataset: ",len(enron_data)

nPOIs = 0
for person in enron_data:
    if enron_data[person]["poi"] == True:
        nPOIs += 1

print "Number of POIs: ",nPOIs

nSalary = 0
for person in enron_data:
    if  enron_data[person]['salary'] != 'NaN':
        nSalary += 1

print "Persons with salaries: ",nSalary

nEmail = 0
for person in enron_data:
    if  enron_data[person]['email_address'] != 'NaN':
        nEmail += 1

print "Persons with emails: ",nEmail

nTotalPay = 0
for person in enron_data:
    if  enron_data[person]['total_payments'] != 'NaN':
        nTotalPay += 1

print "Persons with total pay: ",nTotalPay," (",float(nTotalPay)/float(len(enron_data))*100.,"%)"

nPOIsTotalPay = 0
for person in enron_data:
    if  enron_data[person]['total_payments'] != 'NaN' and enron_data[person]['poi'] == True:
        nPOIsTotalPay += 1

print "POIs with total pay: ",nPOIsTotalPay," (",float(nPOIsTotalPay)/float(nPOIs)*100.,"%)"

#for key,value in sorted(enron_data.items()):
#    print key

print nPOIs," number of POIs in the dataset"

print enron_data["PRENTICE JAMES"]['total_stock_value']
print enron_data["Colwell Wesley".upper()]['from_this_person_to_poi']
print "Skilling: excercised $",enron_data["Skilling Jeffrey k".upper()]['exercised_stock_options'],
print "total $",enron_data["Skilling Jeffrey k".upper()]['total_stock_value']
print "Fastow: excercised $",enron_data["Fastow Andrew S".upper()]['exercised_stock_options'],
print "total $",enron_data["Fastow Andrew S".upper()]['total_stock_value']
print "Lay: excercised $",enron_data["Lay Kenneth L".upper()]['exercised_stock_options'],
print "total $",enron_data["Lay Kenneth L".upper()]['total_stock_value']

print enron_data["PRENTICE JAMES"]
# Andrew Fastow - Chief Financial Officer
# Jeffrey Skilling - CEO
# Kenneth Lay - Chairman of the Enron board of directors
