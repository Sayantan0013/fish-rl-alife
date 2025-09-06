# %% CELL 1
import matplotlib.pyplot as plt
from utils.loggers import load_hdf5_log
from sklearn.decomposition import PCA
import pickle
import numpy as np

# %% CELL


file_path = "dumps/2025-09-06_15-53-22.h5"

all_data = load_hdf5_log(file_path)

def flatten_data(data):
    flattened = {}
    for run, steps in data.items():
        for step, values in steps.items():
            for key, value in values.items():
                try:
                    flattened[key] = flattened.get(key,[]) + value.tolist()
                except:
                    flattened[key] = flattened.get(key,[]) + [value]

    for key, value in flattened.items():
        flattened[key] = np.array(value)

    return flattened

acc_data = flatten_data(all_data)
# %% CELL

observations = acc_data["obs"].reshape(*acc_data['key'].shape[:-1],-1)
observations = observations.reshape(-1, observations.shape[-1])
seen = np.array(observations[:,2:].sum(axis=-1) > 1,dtype=np.int8)
print(seen.shape)

color_map = np.array(['red','blue'])
colors = color_map[seen]

attr = "value"
data = acc_data[attr]

cumulted_messages = np.array(data).reshape(-1, data.shape[-1])

# print(cumulted_messages.min(),cumulted_messages.max())

# Initialize and fit PCA model
pca = PCA(n_components=2)
principal_components = pca.fit_transform(cumulted_messages)

save_root = 'models/embeddings'
np.save(f'{save_root}/data/{attr}.npy', principal_components)
np.save(f'{save_root}/raw_data/{attr}.npy', cumulted_messages)



# Define a filename for the trained PCA model
pca_model_filename =f'{save_root}/{attr}.pkl'

# Save the trained PCA model to a file

pickle.dump(pca, open(pca_model_filename, 'wb'))
print(f"PCA model saved to {pca_model_filename}")

pca = pickle.load(open(pca_model_filename,'rb'))

print(f"Shape of principal components: {principal_components.shape}")
plt.figure(figsize=(10, 8))

colors = color_map[seen]

seen_components = principal_components[seen.astype(np.bool)]
unseen_components = principal_components[~seen.astype(np.bool)]

# plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5, c=colors)

# plt.scatter(seen_components[:, 0], seen_components[:, 1], alpha=0.5, c=colors[seen.astype(np.bool)])
plt.scatter(unseen_components[:, 0], unseen_components[:, 1], alpha=0.5, c=colors[~seen.astype(np.bool)])


plt.show()
# %% CELL


run_id = 3
agent_id = 0

def get_run_data(data, run_id, agent_id = None):
    steps = data[f'run_{run_id}']
    flattened = {}
    for step, values in steps.items():
        for key, value in values.items():
            if agent_id is not None:
                try:
                    flattened[key] = flattened.get(key,[]) + value[:,agent_id].tolist()
                except:
                    flattened[key] = flattened.get(key,[]) + [value]
            else:
                flattened[key] = flattened.get(key,[]) + value.tolist()

    for key, value in flattened.items():
        flattened[key] = np.array(value)

    return flattened


run_data = get_run_data(all_data, run_id, agent_id)

agent_i = pca.transform(run_data[attr])

print(f"Shape of principal components: {principal_components.shape}")
plt.figure(figsize=(10, 8))

plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.2)
plt.plot(agent_i[:, 0], agent_i[:, 1], 'ok', alpha=0.8)

plt.show()
