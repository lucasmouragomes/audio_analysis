import pandas as pd
import numpy as np
import librosa

class AudioData():
    
    def __init__(self, X, Y, df, raw):
        X = X.copy()
        Y = Y.copy()
        df = df.copy()
        raw = raw.copy()
        self.n_mels = X['melspectrogram'][0].shape[0]
        
        X = self.correctShape(X)
        
        shape_dict = {}
        for i in X.columns:
            shape_dict[i] = []
            for j in X[i]:
                shape_dict[i].append(j.shape[1])

        preview_sizes = pd.Series(shape_dict['chroma']).value_counts()
        self.preview_sizes = preview_sizes
        for i in preview_sizes[1:].index.values:
            while True:
                try:
                    delete_index = shape_dict['chroma'].index(i)
                    X.drop(delete_index, inplace=True)
                    Y.drop(delete_index, inplace=True)
                    df.drop(delete_index, inplace=True)
                    raw.drop(delete_index, inplace=True)
                    shape_dict['chroma'][delete_index] = None
                except:
                    break
        
        X.reset_index(drop=True, inplace=True)
        Y.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        raw.reset_index(drop=True, inplace=True)
        
        X = self.calculate_mel(X, preview_sizes)
        
        single_size = [i for i in X.columns if i not in ['melspectrogram', 'mfcc']]
        
        self.X = X
        self.Y = Y
        self.df = df
        self.raw = raw
        

        i = single_size[0]
        X_alt = pd.DataFrame(X[i].apply(self.obs_info).tolist(), columns=[i+'_mean', i+'_var',i+'_meandif',i+'_vardif'])
        for i in single_size[1:]:
            X_alt = X_alt.join(
                pd.DataFrame(X[i].apply(self.obs_info).tolist(), columns=[i+'_mean', i+'_var',i+'_meandif',i+'_vardif'])
                )
            
        self.X_alt = X_alt
        
        train_ind, valid_ind, test_ind = self.split_df(df)

        self.X_train = X_alt.iloc[train_ind]
        self.Y_train = Y.iloc[train_ind]

        self.X_valid = X_alt.iloc[valid_ind]
        self.Y_valid = Y.iloc[valid_ind]

        self.X_test = X_alt.iloc[test_ind]
        self.Y_test = Y.iloc[test_ind]

        X_train_mel = np.dstack(X.loc[train_ind,'melspectrogram'])
        X_valid_mel = np.dstack(X.loc[valid_ind,'melspectrogram'])
        X_test_mel = np.dstack(X.loc[test_ind,'melspectrogram'])

        X_train_mel = np.transpose(X_train_mel, (2, 0, 1))
        X_valid_mel = np.transpose(X_valid_mel, (2, 0, 1))
        X_test_mel = np.transpose(X_test_mel, (2, 0, 1))

        self.X_train_mel = np.expand_dims(X_train_mel, axis=3)
        self.X_valid_mel = np.expand_dims(X_valid_mel, axis=3)
        self.X_test_mel = np.expand_dims(X_test_mel, axis=3)
        
        try:
            corr_analysis = X_alt.join(Y).corr()['valence'].abs().sort_values(ascending=False).drop('valence', axis=0)
        except:
            corr_analysis = X_alt.join(Y).corr()['valence_mean'].abs().sort_values(ascending=False).drop('valence_mean', axis=0)
        self.corr = corr_analysis

        self.train_cols = corr_analysis.head(50).index.values

    def obs_info(self,x):
        return np.array(
            [
            np.mean(x[0]), 
            np.var(x[0]), 
            np.mean(np.diff(x[0], n=1)),
            np.var(np.diff(x[0], n=1))
            ])
        
    def correctShape(self, X): 
        for i in X.columns:
            print(i, 'is shape', X[i].shape)

        X['rmse'] = X['rmse'].apply(lambda x: np.array([x]))

        print('rmse is shape', X['rmse'].shape)
        
        return X
    
    def calculate_mel(self, X, preview_sizes):
        single_size = [i for i in X.columns if i not in ['melspectrogram', 'mfcc']]
        for i in single_size:
            X.loc[:][i] = X[i].apply(lambda x: np.resize(x, (1, preview_sizes.index.values[0])))
        X.loc[:]['melspectrogram'] = X['melspectrogram'].apply(lambda x: librosa.power_to_db(x))
        X.loc[:]['melspectrogram'] = X['melspectrogram'].apply(lambda x: np.resize(x, (self.n_mels, preview_sizes.index.values[0])))
        n_mfccs = X['mfcc'][0].shape[0]
        X.loc[:]['mfcc'] = X['mfcc'].apply(lambda x: np.resize(x,(n_mfccs, preview_sizes.index.values[0])))
        mfcc = X['mfcc'].apply(lambda y: pd.DataFrame(y).apply(lambda x: np.array([x.values]), axis=1))
        mfcc.columns = ['mfcc'+str(i) for i in mfcc.columns]
        X.drop('mfcc', axis=1)
        return X.join(mfcc)

    def split_df(self, df, train=4, valid=1, test=1):
        split_size = int(df.shape[0]/(train+valid+test))
        df = df.sample(frac=1, random_state=42)
        train_df = df[0 : train*split_size].index
        valid_df = df[train*split_size+1 : (train+valid)*split_size].index
        test_df = df[(train+valid)*split_size+1 : df.shape[0]].index
        return train_df, valid_df, test_df