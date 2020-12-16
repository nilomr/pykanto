#%%


from numpy.core.fromnumeric import size
from src.greti.read.paths import DATA_DIR
import pandas as pd
import pickle
import numpy as np
import random
from tqdm.autonotebook import tqdm
from scipy.spatial.distance import cdist, jensenshannon, pdist
from sklearn import mixture
from matplotlib import pyplot as plt
import seaborn as sns

%load_ext autoreload
%autoreload 2


# %%
# Import data

DATASET_ID = "GRETI_HQ_2020_segmented"

dfs_dir = DATA_DIR / "indv_dfs" / DATASET_ID
# indv_dfs = pd.read_pickle(dfs_dir / (f"{DATASET_ID}_labelled_checked.pickle")) #! Change to this once manual check done

indv_dfs = pd.read_pickle('/media/nilomr/SONGDATA/syllable_dfs/GRETI_HQ_2020_segmented/GRETI_HQ_2020_segmented_with_labels.pickle')
indv_dfs = {indv : indv_dfs[indv_dfs.indv == indv] for indv in indv_dfs.indv.unique()}

indvs = list(indv_dfs.keys())

def norm(a):
    """normalizes a string by its average and sd"""
    a = (np.array(a) - np.average(a)) / np.std(a)
    return a


# %%

# labs = "hdbscan_labels_fixed"
labs = "hdbscan_labels"


# %%

indv_1 ='EX76'
indv_2 ='SW5'

def pseudo_psd(spectrogram):
    psd = np.mean(spec, axis=1)
    psd = (psd - psd.min()) / (psd - psd.min()).sum()
    return psd


#%%

bird_list = []
note_specs = []
for indv in tqdm(indvs):
    for lab in indv_dfs[indv][labs].unique():
        if lab > -1:
            n_specs = len(indv_dfs[indv][indv_dfs[indv][labs] == lab].spectrogram)
            for i in random.sample(range(n_specs), 1):
                spec = indv_dfs[indv][indv_dfs[indv][labs] == lab].spectrogram.iloc[i]
                note_specs.append(pseudo_psd(spec))
                bird_list.append(indv)

        
m = cdist(note_specs,note_specs,'euclidean')
dist_df = pd.DataFrame(np.tril(m, k=-1), columns=bird_list, index=bird_list)
y = list(dist_df.stack())


coords = [list(nestboxes[nestboxes['nestbox'] == bird].east_north.values[0]) for bird in bird_list]
spatial_dist = cdist(coords,coords)
spatial_dist_df = pd.DataFrame(np.tril(spatial_dist, k=-1), columns=bird_list, index=bird_list)
x = list(spatial_dist_df.stack())

#%%
df = pd.DataFrame({'s_dist':x,'a_dist':y})

df = df[df['a_dist'] != 0]
#df = df[df['s_dist'] < 600]
df = df[df['s_dist'] != 0]

fig_dims = (8, 4)
fig, ax = plt.subplots(figsize=fig_dims)

sns.regplot(x='s_dist', y='a_dist', data = df, marker='o', scatter_kws={'s':2, 'alpha':0.02})

# %%



# Get basis syllable set
pop_specs = []
pop_nlabs = 0
for indv in tqdm(indvs):
    for lab in indv_dfs[indv][labs].unique():
        if lab > -1:
            pop_nlabs += 1
            n_specs = len(indv_dfs[indv][indv_dfs[indv][labs] == lab].spectrogram)
            for i in random.sample(range(n_specs), 1):
                try:
                    spec = indv_dfs[indv][indv_dfs[indv][labs] == lab].spectrogram.iloc[i]
                    pop_specs.append(pseudo_psd(spec))
                except:
                    print(indv, lab, n_specs) # skip if fewer notes per category than 'needed' - this is provisional, need to infer an adequate sample size and quantify uncertaint
#%%

# 
every_matrix = {}
for index, indv in enumerate(indvs):
    all_psds = []

    for lab in tqdm(indv_dfs[indv][labs].unique()):
        specs = []
        if lab > -1:
            n_specs = len(indv_dfs[indv][indv_dfs[indv][labs] == lab].spectrogram)
            n_specs = n_specs if n_specs < 100 else 100
            for i in range(n_specs):
                spec = indv_dfs[indv][indv_dfs[indv][labs] == lab].spectrogram.iloc[i]
                specs.append(spec)
            all_psds.append(pseudo_psd(sum([i for i in specs]) / n_specs))

    d = cdist(all_psds,pop_specs,'euclidean')
    every_matrix[indv] = 1-(d/np.max(d))


matrix = np.zeros((len(indvs), len(indvs)))

for index, indv_1 in enumerate(indvs):
    for index2, indv_2 in enumerate(indvs):
        if indv_1 == indv_2:
            matrix[index,index] = 0
        elif matrix[index,index2] == 0 and matrix[index,index2] == 0:
            d = cdist(every_matrix[indv_1],every_matrix[indv_2],'euclidean')
            d = 1-(d/np.max(d))
            mean_similarity = np.mean(d) #/ (len(every_matrix[indv_1]) + len(every_matrix[indv_2]))
            matrix[index,index2] = mean_similarity
            matrix[index2,index] = mean_similarity


distances_df = pd.DataFrame(np.tril(matrix, k=-1), columns=indvs, index=indvs)

# %%

# plt.figure(figsize=(14,14))
# sns.heatmap(distances_df)

#%%
# Build matrix of spatial distances

from src.greti.read.paths import RESOURCES_DIR

coords_file = RESOURCES_DIR / "nestboxes" / "nestbox_coords.csv"
tmpl = pd.read_csv(coords_file)

nestboxes = tmpl[tmpl["nestbox"].isin(indvs)]
nestboxes["east_north"] = nestboxes[["x", "y"]].apply(tuple, axis=1)

# %%

spatial_distance = cdist([list(i) for i in nestboxes['east_north']],[list(i) for i in nestboxes['east_north']])
spatial_distance_df = pd.DataFrame(np.tril(spatial_distance, k=-1), columns=nestboxes.nestbox, index=nestboxes.nestbox)[indvs].reindex(indvs)

#%%
#

x = list(spatial_distance_df.stack())
y = list(distances_df.stack())
#x = [i for i in x if i != 0]
#y = [i for i in y if i != 0]

df = pd.DataFrame({'s_dist':x,'a_dist':y})

df = df[df['a_dist'] != 0]

df = df[df['s_dist'] < 600]
df = df[df['s_dist'] != 0]

#%%


# sns.regplot(x='s_dist', y='a_dist', data = df, marker='o', scatter_kws={'s':2, 'alpha':0.03})
# sns.regplot(x='s_dist', y='a_dist', data = df, x_bins=6)
# sns.regplot(x='s_dist', y='a_dist', data = df, x_estimator=np.mean, logx=True, scatter = False)

fig_dims = (8, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.regplot(x='s_dist', y='a_dist', data = df, marker='o', scatter_kws={'s':7, 'alpha':0.7})

#ax.set(yscale="log")


#sns.regplot(x='s_dist', y='a_dist', data = df, lowess=True, scatter = False, x_bins = 6)



#%%


                
# %%

# Mets method
import time
t0 = time.time()

indvs = indvs[:30]

len(indvs)

#indvs = indvs[:4]

matrix = np.zeros((len(indvs), len(indvs)))

for index, indv_1 in tqdm(enumerate(indvs)):
    for index2, indv_2 in enumerate(indvs):
        if indv_1 == indv_2:
            matrix[index,index] = 0
        elif matrix[index,index2] == 0 and matrix[index,index2] == 0:
            # First individual to compare
            indv_1_even = []
            indv_1_odd = []

            for lab in tqdm(indv_dfs[indv_1][labs].unique()):
                if lab > -1:
                    n_specs = len(indv_dfs[indv_1][indv_dfs[indv_1][labs] == lab].spectrogram)
                    n_specs = n_specs if n_specs < 100 else 100
                    for i in range(n_specs):
                        spec = indv_dfs[indv_1][indv_dfs[indv_1][labs] == lab].spectrogram.iloc[i]
                        if (i % 2) == 0:
                            indv_1_even.append(pseudo_psd(spec))
                        else:
                            indv_1_odd.append(pseudo_psd(spec))

            # Second individual to compare
            indv_2_even = []
            indv_2_odd = []

            for lab in tqdm(indv_dfs[indv_2][labs].unique()):
                if lab > -1:
                    n_specs = len(indv_dfs[indv_2][indv_dfs[indv_2][labs] == lab].spectrogram)
                    n_specs = n_specs if n_specs < 100 else 100
                    for i in range(n_specs):
                        spec = indv_dfs[indv_2][indv_dfs[indv_2][labs] == lab].spectrogram.iloc[i]
                        if (i % 2) == 0:
                            indv_2_even.append(pseudo_psd(spec))
                        else:
                            indv_2_odd.append(pseudo_psd(spec))


                            
            from scipy.spatial import distance

            k1 = len([lab for lab in indv_dfs[indv_1][labs].unique() if lab > -1])
            k2 = len([lab for lab in indv_dfs[indv_2][labs].unique() if lab > -1])
            k = np.max([k1,k2])

            #calculate distance matrices:
            d1=cdist(indv_1_even,pop_specs,'sqeuclidean')
            d1_2=cdist(indv_1_odd,pop_specs,'sqeuclidean')
            d2=cdist(indv_2_even,pop_specs,'sqeuclidean')
            d2_2=cdist(indv_2_odd,pop_specs,'sqeuclidean')

            mx=np.max([np.max(d1),np.max(d2),np.max(d1_2),np.max(d2_2)])

            #convert to similarity matrices:
            s1=1-(d1/mx)
            s1_2=1-(d1_2/mx)
            s2=1-(d2/mx)
            s2_2=1-(d2_2/mx)

            #estimate GMMs:
            mod1=mixture.GaussianMixture(n_components=k1,max_iter=100000,n_init=5,covariance_type='full').fit(s1)
            mod2=mixture.GaussianMixture(n_components=k2,max_iter=100000,n_init=5,covariance_type='full').fit(s2)

            len1, len2 = len(d1), len(s2)

            #calculate likelihoods for held out data:
            score1_1=mod1.score(s1_2)
            score2_1=mod2.score(s1_2)
            score1_2=mod1.score(s2_2)
            score2_2=mod2.score(s2_2)

            #len2=float(len(pop_specs))
            #len1=float(len(pop_specs))

            #calculate song divergence (DKL estimate):
            score1= np.log2(np.e)*((np.mean(score1_1))-(np.mean(score2_1)))
            score2= np.log2(np.e)*((np.mean(score2_2))-(np.mean(score1_2)))

            score1=score1/len1
            score2=score2/len2
            
            matrix[index,index2] = score1
            matrix[index2,index] = score2



            # matrix[index2,index] = score2

            # print(score1, score2)
        else:
            print('this already done')


t1 = time.time()
total = t1-t0

#%%


# %%
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
%matplotlib inline


fig = pyplot.figure()
ax = Axes3D(fig)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
labels = mod2.predict(s2)

ax.scatter(s2[:, 5], s2[:, 7], s2[:, 40], c=labels, s=40, cmap='viridis')
pyplot.show()

# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)

#%%

from scipy import linalg
import itertools
import matplotlib as mpl
colours = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
    "#fc6c62",
    "#7c7cc4",
    "#57b6bd",
    "#e0b255",
]
color_iter = itertools.cycle(colours)

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .001, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)



plot_results(s2, mod2.predict(s2), mod2.means_, mod2.covariances_, 0,
             'Gaussian Mixture')

#%%




psd3 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 9].spectrogram.iloc[1], axis=1)
psd4 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 9].spectrogram.iloc[3], axis=1)

set2 = [psd3,psd4]

d1=spatial.distance.cdist(indv_specs, pop_specs, 'sqeuclidean')

#sns.heatmap(1-(d1/np.max(d1)))

#%%
x = np.array(list(indv_dfs[indv]["umap_viz"].values))[:, 0]
y = np.array(list(indv_dfs[indv]["umap_viz"].values))[:, 1]
plt.scatter(x, y, s=10, linewidth=0, alpha=0.3)
plt.show()

# %%
colours = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
    "#fc6c62",
    "#7c7cc4",
    "#57b6bd",
    "#e0b255",
]

for lab, colour in zip(indv_dfs['EX76'].hdbscan_labels_fixed.unique() > -1, colours):
    print(colour)

    for i in range(1, len(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == lab].spectrogram)):
        plt.plot(np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == lab].spectrogram.iloc[i], axis=1), color = colour, alpha =0.03 )
        print()
# %%




# %%

from scipy import spatial
psd1 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 3].spectrogram.iloc[0], axis=1)
psd2 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 5].spectrogram.iloc[0], axis=1)
psd3 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 4].spectrogram.iloc[0], axis=1)
psd4 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 7].spectrogram.iloc[0], axis=1)

set1 = [psd1,psd2, psd3, psd4]

psd3 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 5].spectrogram.iloc[1], axis=1)
psd4 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 5].spectrogram.iloc[2], axis=1)
psd5 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 5].spectrogram.iloc[5], axis=1)
psd6 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 5].spectrogram.iloc[50], axis=1)
psd7 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 5].spectrogram.iloc[20], axis=1)
psd8 = np.mean(indv_dfs['EX76'][indv_dfs['EX76']["hdbscan_labels_fixed"] == 5].spectrogram.iloc[30], axis=1)

set2 = [psd3,psd4,psd5,psd6,psd7,psd8]

d1=spatial.distance.cdist(set1,set2,'sqeuclidean')
sns.heatmap(1-(d1/np.max(d1)))


# %%




import numpy as np
from matplotlib import pyplot as plt

# Seed the random number generator
np.random.seed(0)

time_step = .01
time_vec = np.arange(0, 70, time_step)

# A signal with a small frequency chirp
sig = np.sin(0.5 * np.pi * time_vec * (1 + .1 * time_vec))

plt.figure(figsize=(8, 5))
plt.plot(time_vec, sig)



from scipy import signal
freqs, times, spectrogram = signal.spectrogram(sig)

plt.figure(figsize=(5, 4))
plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')
plt.tight_layout()



freqs, psd = signal.welch(sig)

plt.figure(figsize=(5, 4))
plt.semilogx(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()



plt.show()
