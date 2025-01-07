import json


def load_sugarcrepe(folder_path):
    """load sugarcrepe dataset from local"""

    if folder_path[-1] != '/':
        folder_path += '/'

    # add attribute
    with open (folder_path+'add_att.json', encoding='utf8') as f:
        add_attribute = json.load(f)

    # add object
    with open (folder_path+'add_obj.json', encoding='utf8') as f:
        add_object = json.load(f)

    # replace attribute
    with open (folder_path+'replace_att.json', encoding='utf8') as f:
        replace_attribute = json.load(f)

    # replace object
    with open (folder_path+'replace_obj.json', encoding='utf8') as f:
        replace_object = json.load(f)

    # replace relation
    with open (folder_path+'replace_rel.json', encoding='utf8') as f:
        replace_relation = json.load(f)

    # swap attribute
    with open (folder_path+'swap_att.json', encoding='utf8') as f:
        swap_attribute = json.load(f)

    # swap object
    with open (folder_path+'swap_obj.json', encoding='utf8') as f:
        swap_object = json.load(f)

    # collate together
    dataset = {
        'add_attribute': add_attribute, 'add_object': add_object, 
        'replace_attribute': replace_attribute, 'replace_object': replace_object, 
        'replace_relation': replace_relation, 'swap_attribute': swap_attribute, 
        'swap_object': swap_object,
    }

    return dataset


def coco_cats(coco, image_id):
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)
    cat_ids = [ann['category_id'] for ann in annotations]
    cats = [coco.loadCats(i)[0]['name'] for i in cat_ids]
    return cats


def coco_object_mask(coco, image_id):
    token_to_mask = {}
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)
    for ann in annotations:
        mask = coco.annToMask(ann)
        cat_id = ann['category_id']
        cat = coco.loadCats(cat_id)[0]['name']
        token_to_mask[cat] = mask
    return token_to_mask
        


def filter_sugarcrepe_distict_objects(coco, image_ids):
    """only consider images which has at most one object per annotation category"""
    filtered_image_ids = []
    for image_id in image_ids:
        cats = coco_cats(coco, image_id)
        if len(cats) > 0 and len(cats) == len(set(cats)): 
            filtered_image_ids.append(image_id)
    return filtered_image_ids
