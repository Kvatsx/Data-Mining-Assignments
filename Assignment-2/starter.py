
'''
Created on Sept 6, 2019
@author: Dhruv Bhagat
'''

import arff


name='Data/training_subsetD.arff'
#name='testingD.arff'

file=open(name)
dataset = arff.load(file)

attributes_list = dataset.get("attributes",[])
data=dataset.get("data")



attribute_vector=[]
X=[]


#attribute_info is in the form of (ATTRIBUTE NAME,[POSSIBLE Categories under this ATTRIBUTE])
for idx, attribute_info in enumerate(attributes_list):
	#print(attribute_info)
	attribute_vector+=[attribute_info[0]]	


num_attributes=len(attribute_vector)


print("Last Column is the prediction(Ground Truth), attribute name is = ", attribute_vector[274])
print("Possible Values for this attribute: ",attributes_list[274][1])
print(num_attributes)



for row in data:
	#print(row)
	X+=[row]



print("Length",len(X[0]))

#Exploring the 265th data attribute
print(attributes_list[265][0],attributes_list[265][1])
print("Value of 265th attribute of 0th Data Point: ", X[0][0])

#Exploring the last attribute which is the ground truth class -- True/False
print(attributes_list[274][0],attributes_list[274][1])
print("Ground truth value of 0th Data Point: ", X[0][274])
