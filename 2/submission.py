import pickle
import torch
import pandas as pd
from joblib import dump, load
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor

class Submission:
    def __init__(self):
        self.message = "Submission"

    def submit(self):
        outfile = 'submission.csv'

        output_file = open(outfile, 'w')

        titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
                 'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']
        preds = []

        # test_data = torch.load('./test/test/testX.pt')
        # file_ids = test_data[-1]
        # rgb_data = test_data[0]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())
        torch.cuda.empty_cache()

        read_prep_data = load('preprocessed_testX.joblib')
        te_data = read_prep_data[0]
        file_ids = read_prep_data[1]
        file_ids = [int(numeric_string) for numeric_string in file_ids]
        print('finished load data')

        predictions = np.zeros(shape=(te_data.shape[0], 12))
        dataset_test = TensorDataset(Tensor(te_data), Tensor(predictions))
        batch_size = 8
        test_dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)
        print('finished prep test data')

        model = torch.jit.load('res50_pretrained_model.pt')
        model.eval()
        print('finished load model')

        for batch_inx, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            test_pred = model(x)
            predictions[batch_inx*batch_size:(batch_inx+1)*batch_size, :] = test_pred.detach().cpu().numpy()
            if batch_inx % 5 == 0:
                print('Batch Index: ' + str(batch_inx))

        # for i, data in enumerate(rgb_data):
        #     # Please remember to modify this loop, input and output based on your model/architecture
        #     output = model(data[:1, :, :, :].to('cuda'))
        #     preds.append(output[0].cpu().detach().numpy())
        # print(file_ids.shape)
        # print(predictions.shape)
        print('finished predict')
        df = pd.concat([pd.DataFrame(file_ids), pd.DataFrame(predictions)], axis=1, names=titles)
        df.columns = titles
        df.to_csv(outfile, index=False)
        print("Written to csv file {}".format(outfile))
        print('all finished')
        return df