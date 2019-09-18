import matplotlib.pyplot as plt
import numpy as np
from experiment_utils.plot_utils import load_exps_data, defaultdict, group_by, correct_limit, filter

plt.style.use('ggplot')
import matplotlib
matplotlib.use('TkAgg')


SMALL_SIZE = 32
MEDIUM_SIZE = 36
BIGGER_SIZE = 44
BIG_SIZE = 44
LINEWIDTH = 8
MARKER = None
ALPHA=0.15

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIG_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = [
    (255, 127, 14),
    (44, 160, 44),  # GREEN
    (31, 119, 180),
    (214, 39, 40),  # RED
    (0, 0, 0),
    (137, 137, 137), # DARK GREY
    (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)
]
for idx, color in enumerate(colors):
    colors[idx] = '#%02x%02x%02x' % color
COLORS = dict()
LEGEND_ORDER = {'a-mb-mpo-pr2':0,
                }

data_path = './data/corl_data/pr2_data/reach/'

exps_data = load_exps_data([data_path])

sampling_time = {'meta_mb.envs.mb_envs.walker2d.Walker2dEnv': 0.008,
                 'Walker2d': 0.008,
                 'meta_mb.envs.mb_envs.half_cheetah.HalfCheetahEnv': 0.05,
                 'HalfCheetah': 0.05,
                 'meta_mb.envs.mb_envs.ant.AntEnv': 0.05,
                 'Ant': 0.05,
                 'meta_mb.envs.mb_envs.hopper.HopperEnv': 0.008,
                 'meta_mb.envs.mujoco.hopper_env.HopperEnv': 0.008,
                 'Hopper': 0.008,
                 }
round_plot = {'a-mb-mpo': {'PR2ReachEnv': None,
                      },
            'mb-mpo': {'meta_mb.envs.pr2.real_pr2_reach_env.PR2ReachEnv': None,
                      },
            'me-trpo': {'meta_mb.envs.mb_envs.hopper.HopperEnv': 5000,
                        'meta_mb.envs.mb_envs.walker2d.Walker2dEnv': 5000,
                        'meta_mb.envs.mb_envs.ant.AntEnv': 5000,
                        'meta_mb.envs.mb_envs.half_cheetah.HalfCheetahEnv': 5000,
                      },
            'mbmpo': {'meta_mb.envs.mb_envs.hopper.HopperEnv': 5000,
                        'meta_mb.envs.mb_envs.walker2d.Walker2dEnv': 5000,
                        'meta_mb.envs.mb_envs.ant.AntEnv': 5000,
                        'meta_mb.envs.mb_envs.half_cheetah.HalfCheetahEnv': 5000,
                      }

         }

x_limits = {'PR2Reach': [0, 540],
            }


def prepare_data_for_plot(exp_data,
                          y_key=None,
                          sup_y_key=None,
                          round_x=None,
                          add_sampling_time=False):
    x_y_tuples = []
    x_key = 'n_timesteps'
    asynch = exp_data[0]['flat_params'].get('async', False)
    for exp in exp_data:
        if not asynch:
            if sup_y_key is not None:
                assert type(sup_y_key) is list
                for key in sup_y_key:
                    if key in exp['progress'].keys():
                        x_y_tuples.extend(list(zip(exp['progress'][x_key], exp['progress'][key])))
                    break
            else:
                x_y_tuples.extend(list(zip(exp['progress'][x_key], exp['progress'][y_key])))
        else:
            if sup_y_key is not None:
                assert type(sup_y_key) is list
                for key in sup_y_key:
                    if key in exp['progress'].keys():
                        if 'Data-TimeSoFar' in exp['progress'].keys():
                            x_y_tuples.extend(list(zip(exp['progress']['Data-TimeSoFar'], exp['progress'][key]/3 * 100)))
                        break
            else:
                raise NotImplementedError

    x_y_dict = defaultdict(list)
    try:
        env = exp_data[0]['flat_params']['env']
    except KeyError:
        env = exp_data[0]['flat_params']['env.$class']

    algo = exp_data[0]['flat_params']['algo']
    _round_x = round_plot.get(algo, dict()).get(env, None)
    round_x = _round_x if _round_x is not None else round_x
    for k, v in x_y_tuples:
        if round_x is not None:
            x_y_dict[(k//round_x) * round_x].append(v)
        else:
            x_y_dict[k].append(v)
    means, stddevs = [], []
    for key in sorted(x_y_dict.keys()):
        means.append(np.mean(x_y_dict[key]))
        stddevs.append(np.std(x_y_dict[key]))
    return np.array(sorted(x_y_dict.keys())), np.array(means), np.array(stddevs), x_limits[env]


def sorting_legend(label):
    return LEGEND_ORDER[label]


def get_color(label):
    asynch = True if label[:2] == 'a-' else False
    if label not in COLORS.keys():
        new_color = colors.pop(0)
        COLORS[label] = new_color
        if asynch:
            COLORS[label[2:]] = new_color
        else:
            COLORS['a-' + label] = new_color
    return COLORS[label]


def get_linestyle(label):
    if label.startswith('a'):
        return '-'
    elif label.startswith('m'):
        return '--'
    else:
        return ':'


def plot_from_exps(exp_data,
                   filters={},
                   split_figures_by=None,
                   split_plots_by=None,
                   x_key='n_timesteps',
                   y_key=None,
                   sup_y_key=None,
                   plot_name='./bad-models.png',
                   subfigure_titles=None,
                   plot_labels=None,
                   x_label=None,
                   y_label=None,
                   num_rows=1,
                   y_limits=None,
                   report_max_performance=False,
                   log_scale=False,
                   round_x=None,
                   ):

    exp_data = filter(exp_data, filters=filters)
    exps_per_plot = group_by(exp_data, group_by_key=split_figures_by)
    num_columns = len(exps_per_plot.keys())
    assert num_columns % num_rows == 0
    num_columns = num_columns // num_rows
    fig, axarr = plt.subplots(num_rows, num_columns, figsize=(14, 12))
    axarr = np.reshape(axarr, (num_rows, num_columns))
    fig.tight_layout(pad=5.0, w_pad=0, h_pad=2, rect=[0, 0, 1, 1])

    # iterate over subfigures
    for i, (default_plot_title, plot_exps) in enumerate(sorted(exps_per_plot.items())):
        plots_in_figure_exps = group_by(plot_exps, split_plots_by)
        subfigure_title = subfigure_titles[i] if subfigure_titles else default_plot_title
        r, c = i//num_columns, i%num_columns
        axarr[r, c].set_title(subfigure_title)
        axarr[r, c].xaxis.set_major_locator(plt.MaxNLocator(5))
        axarr[r, c].ticklabel_format(axis='x', )

        # iterate over plots in figure
        y_max_mean = -1e10
        y_axis_min = 1e10
        y_axis_max = -1e10
        for j, default_label in enumerate(sorted(plots_in_figure_exps, key=sorting_legend)):
            exps = plots_in_figure_exps[default_label]
            x, y_mean, y_std, x_limits = prepare_data_for_plot(exps,
                                                     y_key=y_key,
                                                     sup_y_key=sup_y_key,
                                                     round_x=round_x)
            print(subfigure_title)
            print(x)
            print(y_mean)

            label = plot_labels[j] if plot_labels else default_label
            _label = label if i == 0 else "__nolabel__"
            if log_scale:
                axarr[r, c].semilogx(x, y_mean, label=_label, linewidth=LINEWIDTH,
                                     color=get_color(label), linestyle=get_linestyle(label))
            else:
                axarr[r, c].plot(x, y_mean, label=_label, linewidth=LINEWIDTH,
                                 color=get_color(label), linestyle=get_linestyle(label))

            axarr[r, c].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2, color=get_color(label))

            # axis labels
            axarr[r, c].set_xlabel(x_label if x_label else x_key)
            axarr[r, c].set_ylabel(y_label if y_label else y_key)
            if x_limits is not None:
                axarr[r, c].set_xlim(*x_limits)
            if y_limits is not None:
                axarr[r, c].set_ylim(*y_limits)
            else:
                try:
                    _y_axis_min, _y_axis_max = correct_limit(axarr[r, c], x, y_mean-y_std, y_mean+y_std)
                    y_axis_max = max(_y_axis_max, y_axis_max)
                    y_axis_min = min(_y_axis_min, y_axis_min)
                    if max(y_mean) > y_max_mean:
                        y_max_mean = max(y_mean)
                except:
                    pass

        if report_max_performance:
            label = 'max' if i == 0 else "__nolabel__"
            axarr[r, c].plot(axarr[r, c].get_xlim(), [y_max_mean]*2, 'k--', label=label)
        if y_limits is None:
            axarr[r, c].set_ylim([y_axis_min, y_axis_max])

    # fig.legend(loc='lower center', ncol=3, bbox_transform=plt.gcf().transFigure)
    fig.savefig(plot_name)


filter_dict = {}

exps_data_filtered = filter(exps_data, filter_dict)


plot_from_exps(exps_data,
               split_figures_by='env.$class',
               split_plots_by='algo',
               y_key='train-AverageReturn',
               x_key='Time',
               filters=filter_dict,
               sup_y_key=['Real-AvgFinalDistance',
                          ],
               # subfigure_titles=['HalfCheetah - output_bias_range [0.0, 0.1]',
               #                  'HalfCheetah - output_bias_range [0.0, 0.5]',
               #                  'HalfCheetah - output_bias_range [0.0, 1.0]'],
               # plot_labels=['ME-MPG', 'ME-TRPO'],
               x_label='Time (s)',
               y_label='Avg Final Distance (cm)',
               plot_name='./pr2_reach_time.pdf',
               num_rows=1,
               report_max_performance=False,
               log_scale=False,
               round_x=10,
               )