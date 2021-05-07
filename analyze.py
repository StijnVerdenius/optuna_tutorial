from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import optuna

DIR = 'testing2.db'

## load
study = optuna.create_study(
    load_if_exists=True,
    study_name=DIR.split(".")[0],
    storage=f"sqlite:///{DIR}",
    direction='maximize'
)  # Create a new study.
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

## plot
plt.scatter(2 ** df.params_batch_size, df.value)
plt.title("batch_size")
plt.ylabel("accuracy")
plt.tight_layout()
plt.xscale("log")
plt.ylim((0.6, 1.05))
plt.grid()
plt.show()

plt.scatter(2 ** df.params_hidden_dim, df.value)
plt.title("hidden_dim")
plt.ylabel("accuracy")
plt.tight_layout()
plt.xscale("log")
plt.ylim((0.6, 1.05))
plt.grid()
plt.show()

plt.scatter(df.params_lr, df.value)
plt.title("learning rate")
plt.ylabel("accuracy")
plt.tight_layout()
plt.xscale("log")
plt.ylim((0.6, 1.05))
plt.grid()
plt.show()

plt.scatter(df.params_weight_decay, df.value)
plt.title("weight_decay")
plt.ylabel("accuracy")
plt.tight_layout()
plt.xscale("log")
plt.ylim((0.6, 1.05))
plt.grid()
plt.show()

plot_dict = defaultdict(list)
for activation in df.params_activation.unique():
    subset = df[df.params_activation == activation]
    mean = subset.value.mean()
    stdev = subset.value.std()
    conf = (2.576 * stdev) / np.sqrt(len(subset))
    plot_dict['mean'].append(mean)
    plot_dict['conf'].append(conf)
    plot_dict['name'].append(activation)
x_pos = np.arange(len(plot_dict["name"]))
plt.bar(x_pos, plot_dict['mean'], yerr=plot_dict['conf'])
plt.xticks(x_pos, plot_dict['name'])
plt.title("activations")
plt.ylim((0.6, 1.05))
plt.grid()
plt.tight_layout()
plt.show()
