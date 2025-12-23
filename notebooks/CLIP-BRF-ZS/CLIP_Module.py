# Calculate the confusion matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import time
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

#####CLIP model#####

def nameCover(Pre_name):
    #return Pre_name
    if "No" not in Pre_name:
        if "Hiking" in Pre_name:
            return "Hiking"

        if "Backpacking" in Pre_name:
            return "Backpacking"
            #return "Hiking"

        if "Boating" in Pre_name:
            return "Boating"    

        if "Swimming" in Pre_name:
            return "Swimming"

        if "Camping" in Pre_name:
            return "Camping"

        if "Fishing" in Pre_name:
            return "Fishing"

        if "Biking" in Pre_name:
            return "Biking"

        if "Horseback_Riding" in Pre_name:
            return "Horseback_Riding"

        if "Wildlife_Viewing" in Pre_name:
            return "Wildlife_Viewing"

        if "Shelling" in Pre_name:
            return "Shelling" 

        if "Surfing" in Pre_name:
            return "Surfing"

        if "Hunting" in Pre_name:
            return "Hunting"

        if "Landscape_Aesthetics" in Pre_name:
            return "Landscape_Aesthetics"    
    
    if Pre_name == "No_CES":
        return "No_CES"
    
    return "No_CES" #Others

# def nameCover(Pre_name):
#     if Pre_name == "Hiking":
#         return "Hiking"
    
#     if Pre_name == "Backpacking":
#         return "Backpacking"
#         #return "Hiking"
    
#     if Pre_name == "Boating":
#         return "Boating"    
    
#     if Pre_name == "Swimming":
#         return "Swimming"
    
#     if Pre_name == "Camping":
#         return "Camping"
    
#     if Pre_name == "Fishing":
#         return "Fishing"
    
#     if Pre_name == "Biking":
#         return "Biking"
    
#     if Pre_name == "Horseback_Riding":
#         return "Horseback_Riding"
    
#     if Pre_name == "Wildlife_Viewing":
#         return "Wildlife_Viewing"
     
#     if Pre_name == "Shelling":
#         return "Shelling" 
    
#     if Pre_name == "Surfing":
#         return "Surfing"
    
#     if Pre_name == "Hunting":
#         return "Hunting"
    
#     if Pre_name == "Landscape_Aesthetics":
#         return "Landscape_Aesthetics"    
    
#     if Pre_name == "No_CES":
#         return "No_CES" 

def classification_metrics(descriptions, conf_matrix, class_names): #Metrics calcuation.
    # Compute accuracy
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    
    # True positives, false positives, and false negatives
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    
    # Precision, recall, and F1-score per class
    precision = np.divide(TP, TP + FP, where=(TP + FP) > 0, out=np.zeros_like(TP, dtype=float))
    recall = np.divide(TP, TP + FN, where=(TP + FN) > 0, out=np.zeros_like(TP, dtype=float))
    f1score = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) > 0, out=np.zeros_like(TP, dtype=float))
    
    # Macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1score)
    
    HashRes = {}
    for i in range(len(class_names)):
        cla = class_names[i]
        pre = round(precision[i]*100)
        rec = round(recall[i]*100)
        f1 = round(f1score[i]*100)
        print(f"{cla}, Precision:{pre}, Recall:{rec}, F1:{f1}")
        HashRes[cla] = {
                            #'Description': descriptions[cla],
                            'Description': cla,
                            'Precision':pre,
                            'Recall':rec,
                            'F1-Score':f1,
                        }
    acc = round(accuracy*100)
    avgPre = round(precision_macro*100)
    avgRec = round(recall_macro*100)
    avgF1 = round(f1_macro*100)
    print("Overall Acc: ", acc)
    print("Overall Precision: ", avgPre)
    print("Overall Recall: ", avgRec)
    print("Overall F1: ",avgF1)

    HashRes['Overall'] = {
        "acc":acc,
        "avgPre":avgPre,
        "avgRec":avgRec,
        "avgF1":avgF1}
    
    return HashRes

def plotCom(label, pred, colab, save_dir, SaveName = f"{''}-{''}-ModelComparsion4RecCNN", show=True): #Draw confusion matrix.
    os.makedirs(f'{save_dir}/PredRes/', exist_ok=True)
    
    #colab = np.unique(colab)
    cm_nor = confusion_matrix(label, pred, labels=colab, normalize='true')*100

    # Normalize the confusion matrix by row
    cm_normalized_int_array = np.round(cm_nor, 0).astype(int)
    cm_normalized_int_array_checking = np.round(cm_nor, 1)
    print(cm_normalized_int_array_checking)

    #fig, ax = plt.subplots()  # Use fig, ax to properly handle plot saving later
    fig, ax = plt.subplots(figsize=(7, 7))  # Set figure size (width=10, height=8)
    # Plot the normalized confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized_int_array, display_labels=colab)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    #plt.title("CLIP")
    plt.xticks(rotation=90)
    if show: plt.show()

    # Calculate the accuracy
    accuracy = accuracy_score(label, pred)
    disp.figure_.savefig(f"{save_dir}/PredRes/{SaveName}_NorConfusionNormMatrix_Acc_{accuracy:.2f}.png",dpi=300, bbox_inches='tight')


    #fig, ax = plt.subplots()  # Use fig, ax to properly handle plot saving later
    fig, ax = plt.subplots(figsize=(7, 7))  # Set figure size (width=10, height=8)
    # Plot the confusion matrix
    cm = confusion_matrix(label, pred, labels=colab)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=colab)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    #plt.title("CLIP")
    plt.xticks(rotation=90)
    if show: plt.show()
    disp.figure_.savefig(f"{save_dir}/PredRes/{SaveName}_ConfusionNormMatrix_Acc_{accuracy:.2f}.png",dpi=300, bbox_inches='tight')
    print(f"Accuracy: {accuracy*100:0.0f}")
    
    return cm, accuracy


def storeImgs(promptHikingHash, preprocess, images_paths_map): #Store the images.
    original_images = []
    images = []
    images_class = []
    
    for k, v in promptHikingHash.items():
        idx = k.split("_")[0]
        image_path = images_paths_map.get(str(idx)) 
        if image_path:
            image = Image.open(image_path).convert("RGB")
            original_images.append(image)
            #images.append(preprocess(image).to(device))
            images.append(preprocess(image).to("cpu"))
            images_class.append(nameCover(v))
    
    return images, images_class

def storeImgsAllFL(promptHikingHash, preprocess, images_paths_map): #Store the images.
    original_images = []
    images = []
    
    for k, v in promptHikingHash.items():
        idx = k.split("_")[0]
        image_path = images_paths_map.get(str(idx)) 
        if image_path:
            image = Image.open(image_path).convert("RGB")
            original_images.append(image)
            images.append(preprocess(image).to(device))
    
    return images


def CLIP(descriptions, images, class_names, model, tokenizer): #Clip operation.
    print('start')
    import psutil
    print('start')
    # Get the memory usage in bytes
    memory_usage = psutil.Process().memory_info().rss
    
    texts = []
    for name in class_names:
        texts.append(descriptions[name])   
    torch.cuda.empty_cache()  # Clear unused memory
    
    with torch.no_grad():
        image_input = torch.stack(images).to(device)
        text_tokens = tokenizer([desc for desc in texts]).to(device)

        # Start the timer
        start_time = time.perf_counter()

        similarityCol = []
        #for i in range(len(image_subinput)):
        for i in range(int(len(image_input)/1000)+1):
            sub_start_time = time.perf_counter()
            print(i)
            if i == int(len(image_input)/1000):
                image_subinput = image_input[i*1000:,:,:,:]
                #print(len(image_input[i*1000:,:,:,:]),i*1000)           
            else:
                image_subinput = image_input[i*1000:1000+i*1000,:,:,:]
                #print(len(image_input[i*1000:1000+i*1000,:,:,:]), i*1000, 1000+i*1000)

            with torch.no_grad():
                #image_features = model.encode_image(image_subinput[i]).float()
                image_features = model.encode_image(image_subinput).float()
                text_features = model.encode_text(text_tokens).float()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            subsimilarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            similarityCol.append(subsimilarity)

            del image_features, text_features
            torch.cuda.empty_cache()  # Clear unused memory
            
            # Get the memory usage in bytes
            memory_usage = psutil.Process().memory_info().rss

            # Convert to megabytes
            memory_usage_mb = memory_usage / (1024 ** 2)
            print(f"Memory Usage: {memory_usage_mb:.2f} MB")
            
            # End the timer
            sub_end_time = time.perf_counter()
            # Calculate the time taken
            execution_time = sub_end_time - sub_start_time
            print(f"{i} Execution time: {execution_time:.6f} seconds")


        # End the timer
        end_time = time.perf_counter()
        # Calculate the time taken
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")

        similarity = np.hstack(similarityCol)
        similarity.shape

        # Find the index of the highest similarity score for each image
        predicted_indices = np.argmax(similarity, axis=0)

        ## Use these indices to get the corresponding labels
        predicted_labels = [class_names[idx] for idx in predicted_indices]

        predicted_cover_labels = [nameCover(x) for x in predicted_labels]


        # Example similarity matrix
        # Get the top 5 indices in each column
        """
        top_5_indices_per_column = np.argsort(similarity, axis=0)[-5:][::-1]

        # Get the top 5 values in each column
        top_5_values_per_column = np.take_along_axis(similarity, top_5_indices_per_column, axis=0)

        # Create a new matrix with the top 5 class names for each column
        top_5_class_names_matrix = np.empty_like(top_5_indices_per_column, dtype=object)

        for i in range(top_5_indices_per_column.shape[0]):
            for j in range(top_5_indices_per_column.shape[1]):
                top_5_class_names_matrix[i, j] = nameCover(class_names[top_5_indices_per_column[i, j]])
        """
        top_indices_per_column = np.argsort(similarity, axis=0)[::-1]

        # Get the top 5 values in each column
        top_values_per_column = np.take_along_axis(similarity, top_indices_per_column, axis=0)

        # Create a new matrix with the top 5 class names for each column
        top_class_names_matrix = np.empty_like(top_indices_per_column, dtype=object)

        for i in range(top_indices_per_column.shape[0]):
            for j in range(top_indices_per_column.shape[1]):
                top_class_names_matrix[i, j] = nameCover(class_names[top_indices_per_column[i, j]])
        """
        print("Top 5 indices per column:")
        print(top_5_indices_per_column)

        print("Top 5 values per column:")
        print(top_5_values_per_column)

        # Print the matrix
        print("Top 5 class names matrix:")
        print(top_5_class_names_matrix)
        """

        # Add a new column 'pred_res' with default values (e.g., None or any initial value you prefer)
        #return predicted_cover_labels, top_5_class_names_matrix, top_5_values_per_column
        return predicted_cover_labels, top_class_names_matrix, top_values_per_column

def CLIP_MultPrompt(descriptions, images, descriptions_keys_list, class_names, model, tokenizer): #Clip operation.
    print('start')
    import psutil
    print('start')
    
    from collections import Counter
    # Strip trailing digits if present (like 'Hiking1' -> 'Hiking')
    base_keys = [k.rstrip('0123456789_') for k in descriptions_keys_list]

    # Count frequencies
    counts = Counter(base_keys)

    # Wrap in a list as requested
    count_dic = dict(counts)

    # Get the memory usage in bytes
    memory_usage = psutil.Process().memory_info().rss
    
    texts = []
    for name in descriptions_keys_list:
        texts.append(descriptions[name])   
    torch.cuda.empty_cache()  # Clear unused memory
    
    with torch.no_grad():
        image_input = torch.stack(images).to(device)
        text_tokens = tokenizer([desc for desc in texts]).to(device)

        # Start the timer
        start_time = time.perf_counter()

        similarityCol = []
        #for i in range(len(image_subinput)):
        for i in range(int(len(image_input)/1000)+1):
            sub_start_time = time.perf_counter()
            print(i)
            if i == int(len(image_input)/1000):
                image_subinput = image_input[i*1000:,:,:,:]
                #print(len(image_input[i*1000:,:,:,:]),i*1000)           
            else:
                image_subinput = image_input[i*1000:1000+i*1000,:,:,:]
                #print(len(image_input[i*1000:1000+i*1000,:,:,:]), i*1000, 1000+i*1000)

            with torch.no_grad():
                #image_features = model.encode_image(image_subinput[i]).float()
                image_features = model.encode_image(image_subinput).float()
                text_features = model.encode_text(text_tokens).float()
                #print(image_features.shape)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            subsimilarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            similarityCol.append(subsimilarity)

            del image_features, text_features
            torch.cuda.empty_cache()  # Clear unused memory
            
            # Get the memory usage in bytes
            memory_usage = psutil.Process().memory_info().rss

            # Convert to megabytes
            memory_usage_mb = memory_usage / (1024 ** 2)
            print(f"Memory Usage: {memory_usage_mb:.2f} MB")
            
            # End the timer
            sub_end_time = time.perf_counter()
            # Calculate the time taken
            execution_time = sub_end_time - sub_start_time
            print(f"{i} Execution time: {execution_time:.6f} seconds")


        # End the timer
        end_time = time.perf_counter()
        # Calculate the time taken
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")

        similarity = np.hstack(similarityCol)
        similarity.shape
        print(similarity.shape)
        
        similarityAvg = np.zeros((len(class_names), similarity.shape[1]))
        start = 0
        for i, cla in enumerate(class_names):
            count = count_dic[cla]
            end = start + count
            similarityAvg[i, :] = np.mean(similarity[start:end, :], axis=0)
            start = end
            

        # Find the index of the highest similarity score for each image
        #predicted_indices = np.argmax(similarity, axis=0)
        predicted_indices = np.argmax(similarityAvg, axis=0)

        ## Use these indices to get the corresponding labels
        #predicted_labels = [descriptions_keys_list[idx] for idx in predicted_indices]
        predicted_labels = [class_names[idx] for idx in predicted_indices]

        predicted_cover_labels = [nameCover(x) for x in predicted_labels]

        top_indices_per_column_multi = np.argsort(similarity, axis=0)[::-1]
        top_indices_per_column = np.argsort(similarityAvg, axis=0)[::-1]

        # Get the top 5 values in each column
        top_values_per_column_multi = np.take_along_axis(similarity, top_indices_per_column_multi, axis=0)
        top_values_per_column = np.take_along_axis(similarityAvg, top_indices_per_column, axis=0)

        # Create a new matrix with the top 5 class names for each column
        top_class_names_matrix = np.empty_like(top_indices_per_column, dtype=object)
        top_class_names_matrix_multi = np.empty_like(top_indices_per_column_multi, dtype=object)

        for i in range(top_indices_per_column.shape[0]):
            for j in range(top_indices_per_column.shape[1]):
                top_class_names_matrix[i, j] = nameCover(class_names[top_indices_per_column[i, j]])
                #top_class_names_matrix[i, j] = descriptions_keys_list[top_indices_per_column[i, j]]
   
        for i in range(top_indices_per_column_multi.shape[0]):
            for j in range(top_indices_per_column_multi.shape[1]):
                top_class_names_matrix_multi[i, j] = descriptions_keys_list[top_indices_per_column_multi[i, j]]
                
        # Add a new column 'pred_res' with default values (e.g., None or any initial value you prefer)
        #return predicted_cover_labels, top_5_class_names_matrix, top_5_values_per_column
        return predicted_cover_labels, top_class_names_matrix, top_values_per_column, top_class_names_matrix_multi, top_values_per_column_multi   

def hash2pandas(HashRes, name):
    # Convert HashRes dictionary into a DataFrame
    df = pd.DataFrame.from_dict(HashRes, orient='index').reset_index()
    df.rename(columns={'index': 'Activity'}, inplace=True)
    df["Data type"] = name
    # Save the DataFrame to a CSV file
    df.to_csv("hashres_results.csv", index=False)

    return df

def saveImg(dn, ModelComparsion, images_paths_map, save_dir, show=False):
    for i in range(len(ModelComparsion)):
        label = ModelComparsion["label"].iloc[i]
        pred = ModelComparsion["Clip_Predictions"].iloc[i]
        pid = ModelComparsion["id"].iloc[i]
        path = images_paths_map[str(pid)]
        plt.figure(figsize=(5, 10))
        image = Image.open(path).convert("RGB")
        plt.imshow(image)
        top_5_preds = list(ModelComparsion.loc[i,['top1', 'top2', 'top3','top4', 'top5']])
        top_5_preds_str = ', '.join(top_5_preds)
        plt.title(f"Actual:{label}\n Predict:{pred}\nTop5 Predict:{top_5_preds_str}")
        plt.xticks([])
        plt.yticks([])

        savePath = f"{save_dir}/{dn}-PredImgs/{label}/"
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, f"{i}_{pid}_actual_{label}_pred_{pred}_top5Pred_{top_5_preds_str}.png"), bbox_inches='tight')
        
        if not show:
            plt.close()
            

def GetPred(similarity, class_names):
    # Find the index of the highest similarity score for each image
    predicted_indices = np.argmax(similarity, axis=0)

    ## Use these indices to get the corresponding labels
    predicted_labels = [class_names[idx] for idx in predicted_indices]

    predicted_cover_labels = [nameCover(x) for x in predicted_labels]

    # Add a new column 'pred_res' with default values (e.g., None or any initial value you prefer)
    #metadata['Clip_Predictions'] = predicted_cover_labels ##
    
    top_indices_per_column = np.argsort(similarity, axis=0)[::-1]

    # Get the top 5 values in each column
    top_values_per_column = np.take_along_axis(similarity, top_indices_per_column, axis=0)

    # Create a new matrix with the top 5 class names for each column
    top_class_names_matrix = np.empty_like(top_indices_per_column, dtype=object)

    for i in range(top_indices_per_column.shape[0]):
        for j in range(top_indices_per_column.shape[1]):
            top_class_names_matrix[i, j] = nameCover(class_names[top_indices_per_column[i, j]])
            
    tops = np.hstack((np.transpose(top_class_names_matrix),np.transpose(top_values_per_column)))
    
    return predicted_cover_labels, tops