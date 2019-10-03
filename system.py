import sys

if not sys.warnoptions:
    import warnings, os
    warnings.simplefilter("ignore")

import time
import pandas as pd
import user_view, admin_view
data = pd.read_csv("database.csv")
n = len(data)

while True:
	x = input("Enter admin or user (A/U): ")
	if x.lower() == 'a':
		y = input("Enter password")
		if y == 'tequedlabs':
			print("Confirmed")
			z = input("Do you want to add new student samples? (y/n)").lower()
			if z == 'y':
				admin_view.add_new_faces(n)
				admin_view.train(n)
			else:
				continue
		else:
			print("Wrong password")
	elif x.lower() == 'u':
		user_view.recognizer()
		time.sleep(1)
		break
	
	else:
		print("Please give valid input")			
