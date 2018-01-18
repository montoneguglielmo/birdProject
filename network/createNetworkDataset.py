import gzip
import cPickle as pickle
import numpy as np

if __name__ == '__main__':


    f = gzip.open('../dataBirdTrash.pkl.gz', 'rb')
    data = pickle.load(f)
    f.close()

    nameBirdsTrash = ['1296_calls', '1440_calls', '1511_calls', '1513_calls', 'trash']
    
    min_data = np.inf
    for dt in data:
        min_data = min(min_data, len(data[dt]))


    dataset = []
    label   = []
    
    for nm in nameBirdsTrash:
        indx = [nB  in nm for nB in nameBirdsTrash]
        indx = np.where(np.asarray(indx))[0][0]
        for cnt_call in range(min_data):
            call = data[nm][cnt_call]
            call = np.reshape(call, np.prod(call.shape))
            dataset.append(call)
            label.append(indx)


    dataset = np.vstack(dataset)
    label   = np.hstack(label)

    indperm = np.random.permutation(dataset.shape[0])

    dataset = dataset[indperm,:]
    label   = label[indperm]

    
    dataset = (dataset - dataset.min())/(dataset.max() - dataset.min())

    
    with gzip.open('dataSingleBirdTrashNN.pkl.gz', 'wb') as f:
        pickle.dump((dataset, label), f, protocol=pickle.HIGHEST_PROTOCOL)


    label[np.where(label==1)[0]] = 0
    label[np.where(label==2)[0]] = 0
    label[np.where(label==3)[0]] = 0
    label[np.where(label==4)[0]] = 1
        
    with gzip.open('dataBirdTrashNN.pkl.gz', 'wb') as f:
        pickle.dump((dataset, label), f, protocol=pickle.HIGHEST_PROTOCOL)
