
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import os
import torch


from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import cv2


class MyDataset(InMemoryDataset):
    def __init__(self, root, data_list, name='data.pt', transform=None):
        self.data_list = data_list
        self.name=name
        
        file_path = root+"processed/"+name
        if os.path.exists(file_path):
            os.remove(file_path)
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return self.name

    def process(self):
        
        torch.save(self.collate(self.data_list), self.processed_paths[0])


from PIL import Image, ImageDraw

def create_image_from_tensor(data, predicted,gnd_image=None, threshold=0.5):
    # Extract each column of the tensor into a separate variable
    gnd_image=cv2.imread("gnn/fb_gnd_bw.jpg")
    tensor,label, edge_index,edge_attr = data.x,data.y,data.edge_index,data.edge_attr
    
    team0, team1,x, y, bp = tensor[:, 0], tensor[:, 1], tensor[:, 2], tensor[:, 3], tensor[:, 4]

    # Get the dimensions of the ground truth image or create a new image if none is provided
    width, height = 525, 340
    if np.all(gnd_image != None)  :
        #height,width ,_ = gnd_image.shape
        
        image = Image.fromarray(gnd_image)
        image=image.resize((width, height))
    else:
        
        image = Image.new('RGB', (width, height), color='white')

    # Create a new drawing context for the image
    draw = ImageDraw.Draw(image)
    bpID= torch.where(bp==1)[0]
    draw.text((10,10),data.name ,fill=(0,0,0))
    # Iterate over each row in the tensor
    for i in range(tensor.shape[0]):

        # Get the x and y coordinates for the player
        x_val, y_val = int(x[i] * width), int(y[i] * height)

        # Get the team and bp values for the player
        team0_val, team1_val, bp_val = bool(team0[i]), bool(team1[i]), bool(bp[i])

        # Get the predicted and label values for the player
        predicted_val, label_val = predicted[i], label[i]

        

        # Determine the color of the player circle based on their team
        if team0_val:
            color = (0, 0, 255)  # Blue
        elif team1_val:
            color = (255, 0, 0)  # Red
        else:
            color = (255, 255, 255)  # White

        # Draw the player circle
        cirsz=5
        off=50
        if bp_val:
            draw.ellipse((x_val-cirsz*2, y_val-cirsz*2, x_val+cirsz*2, y_val+cirsz*2), outline=color, fill=None)

        draw.ellipse((x_val-cirsz, y_val-cirsz, x_val+cirsz, y_val+cirsz), fill=color)

        # Draw the line if predicted > threshold
        if i==bpID:
            if label_val ==1 :print(231)
            continue
        #draw.text((width-off, height-10-off), "{}/{}".format("pred", "lbl"), fill=(255, 255, 255) )
        if label_val ==1 :
            col=(0,255,0)
            
            # Find the x and y coordinates of the player with bp=True
            bp_x = int(x[bp==1][0] * width)
            bp_y = int(y[bp==1][0] * height)
            # Draw the line and label

            index=-1
            for idx,e in enumerate(edge_index.T):
                if torch.all(e==torch.tensor([bpID,i])):
                    index=idx
                    break
            edge_val = edge_attr[index]
            tx=bp_x + (x_val-bp_x)//2
            ty=bp_y + (y_val-bp_y)//2
            draw.line((x_val, y_val, bp_x, bp_y), fill=(0,255,0))
            draw.text((tx,ty), "{:.2f}".format(float(predicted_val)), fill=col)
            #draw.text((tx,ty), "{:.2f}/{:.0f}".format(float(predicted_val), label_val), fill=(255, 255, 255) )

        
        if predicted_val > threshold :

            # Find the x and y coordinates of the player with bp=True
            bp_x = int(x[bp==1][0] * width)
            bp_y = int(y[bp==1][0] * height)
            # Draw the line and label

            index=-1
            for idx,e in enumerate(edge_index.T):
                if torch.all(e==torch.tensor([bpID,i])):
                    index=idx
                    break
            edge_val = edge_attr[index]
            tx=bp_x + (x_val-bp_x)//2
            ty=bp_y + (y_val-bp_y)//2
            draw.line((x_val, y_val, bp_x, bp_y), fill=(255,255,255))
            if label_val==1:
                col=(0,255,0)
            else:
                col=(255,180,100)
            draw.text((tx,ty), "{:.2f}".format(float(predicted_val)), fill=col)
            

        
    
    return image


