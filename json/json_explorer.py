import json

nuevo = {}
nuevo['categories'] = []
nuevo['categories'].append({
	"supercategory" : "Bottle",
	"id" : 0,
	"name" : "Clear plastic bottle"
})
nuevo['categories'].append({
	"supercategory" : "Paper",
	"id" : 1,
	"name" : "Normal paper"
})
nuevo['annotations'] = []
nuevo['images'] = []
	
# print category names
with open('./taco.json') as json_file:
	data = json.load(json_file)

	# Each entry looks like
	#	supercategory :	string
	#	id : 		in
	#	name : 		string
	#for x in data['categories']:
	#	print("id: " + str(x['id']) + "\tname: " + x['name'] + "\tsuper: " + x['supercategory'])

	img_ids = []

	# 5: Clear plastic bottle
	# 33: Normal paper

	# Each entry looks like
	#	id : 		int
	#	image_id : 	int
	#	category_id :	int
	#	segmentation :	array
	#	area :		int
	#	bbox :		array
	#	iscrowd :	int
	for x in data['annotations']:
		cat = x['category_id']
		if ( cat == 5 or cat == 33 ):
			# add entry to json
			nuevo['annotations'].append(x)
			# add image_id to list, we need to add them to json
			img_ids.append( x['image_id'] )

	# Each entry looks like
	#	id :		int
	#	file_name :	string		# "batch_X/0000XX.jpg"
	#	...
	for x in data['images']:
		if ( x['id'] in img_ids ):
			x['file_name'] = x['file_name'].replace("/", "-")
			nuevo['images'].append(x)

for x in nuevo['annotations']:
	if (x['category_id'] == 5):
		x['category_id'] = 0
	elif (x['category_id'] == 33):
		x['category_id'] = 1
			
#print( json.dumps(nuevo, indent=4) )

with open('taquito.json', 'w') as outfile:
    json.dump(nuevo, outfile)



