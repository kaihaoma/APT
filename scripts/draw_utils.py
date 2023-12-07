import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv
import draw_utils

# color_list = ["r", "g", "c", "m", "k"]
# marker_list = ["o", "x", "v", "*", "^"]

zuo_green = "#5daa2d"
zuo_oriange = "#f2883f"
zuo_blue = "#268be6"
zuo_red = "#fc0464"
zuo_purple = "#7863d6"

color_list = [zuo_blue, zuo_green, zuo_oriange, zuo_purple, zuo_red]
# color_list = "gbcmrw"
marker_list = "oxvD*"
marker_size_list = [10, 10, 10, 10, 15]
linestyle_list = ["--", "--", "--", "--", "-"]


def get_fmt(id):
    return f"{marker_list[id]}{linestyle_list[id]}"


def get_color(id):
    return color_list[id]


def plt_init(figsize=None, labelsize=24, subplot_flag=False):
    if figsize is not None:
        plt.figure(figsize=figsize)

    plt.clf()
    if not subplot_flag:
        ax = plt.gca()
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=labelsize,
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )


def plt_save_and_final(save_path):
    print(f"[Note]Save to {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def read_csv(input_path=None, has_header=True):
    # print(f"[Note]read_csv from {input_path}")
    with open(input_path, "r") as file:
        reader = csv.reader(file)
        if has_header:
            headers = next(reader)
        else:
            headers = None
        # print(f"[Note]headers:{headers}")
        elements = [row for row in reader if len(row) > 0]

    return headers, elements


def plot_line(
    plot_x_list,
    plot_y_list,
    labels,
    scatter_list=None,
    xticks=None,
    yticks=None,
    xlabels=None,
    ylabels=None,
    font_size=24,
    legends_font_size=None,
    save_path=None,
    fig_size=None,
    set_line_id=None,
    axhyline=None,
    **legend_kwargs,
):
    plt_init(figsize=fig_size)
    ax = plt.gca()
    num_lines = len(plot_x_list)
    print("[Note]num_lines:", num_lines)
    for line_id in range(num_lines):
        fmt_id = line_id if set_line_id is None else set_line_id[line_id]
        plt.plot(
            plot_x_list[line_id],
            plot_y_list[line_id],
            get_fmt(fmt_id),
            color=get_color(fmt_id),
            markerfacecolor="none",
            markersize=marker_size_list[fmt_id],
            label=labels[line_id],
        )
    if scatter_list is not None:
        for i in range(len(scatter_list)):
            plt.scatter(scatter_list[i][0], scatter_list[i][1], c="r", marker="*")
    if xticks is not None:
        plt.xticks(xticks, fontsize=font_size)
    else:
        xlims = ax.get_xlim()
        print(f"[Note]xlims:{xlims}")
    if yticks is not None:
        plt.yticks(yticks, fontsize=font_size)
        ax.set_ylim([yticks[0], yticks[-1]])
    else:
        ylims = ax.get_ylim()
        print(f"[Note]ylims:{ylims}")
    if xlabels is not None:
        if "%" in xlabels:
            print(f"[Note]Set PercentFormatter in xaxis")
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.xlabel(xlabels, fontsize=font_size)
    if ylabels is not None:
        if "%" in ylabels:
            print(f"[Note]Set PercentFormatter in yaxis")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.xlabel(xlabels, fontsize=font_size)
        plt.ylabel(ylabels, fontsize=font_size)

    if axhyline is not None:
        plt.axhline(y=axhyline, color="k", linestyle="--")
    if legends_font_size is None:
        legends_font_size = font_size - 6

    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    handles, labels = ax.get_legend_handles_labels()
    ncols = legend_kwargs.get("ncols", 1)
    print(f"[Note]ncols:{ncols}")
    ax.legend(
        flip(handles, ncols),
        flip(labels, ncols),
        fontsize=legends_font_size,
        edgecolor="k",
        **legend_kwargs,
        # ncols=2,
        # loc="upper right",
        # bbox_to_anchor=(0.7, 0.9),
        # columnspacing=1.2,
        # handlelength=1.0,
        # labelspacing=0.3,
    )

    # ax.legend(, ncol=3)
    plt_save_and_final(save_path=save_path)


def plot_simple_line(plot_x_list, plot_y_list, label_list, xticks=None, yticks=None, xlabels=None, ylabels=None, legend=False, save_path=None):
    ax = plt.gca()
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=24,
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    for i, (x, y, label) in enumerate(zip(plot_x_list, plot_y_list, label_list)):
        ax.plot(x, y, get_fmt(i), color=get_color(i), markerfacecolor="none", markersize=marker_size_list[i], label=label)

    if legend:
        ax.legend(
            fontsize=24,
            edgecolor="k",
            ncols=6,
            # loc="upper center",
            # bbox_to_anchor=(-0.1, 1.35),
            # three figures
            # bbox_to_anchor=(0.5, 1.4),
            # two figures
            bbox_to_anchor=(0.9, 1.4),
        )
    if xticks is not None:
        plt.xticks(xticks, fontsize=24)
    if yticks is not None:
        plt.yticks(yticks, fontsize=24)
    if xlabels is not None:
        plt.xlabel(xlabels, fontsize=24)
    if ylabels is not None:
        plt.ylabel(ylabels, fontsize=24)

    if save_path is not None:
        plt_save_and_final(save_path=save_path)


ori_label_list = ["DP", "MP", "SP", "NP"]
map_label_list = ["GDP", "NFP", "SNP", "NFP"]


# labels for whole subgraph
def plot_main_exp(elements, label_list, xlabels=None, ylabels=None, subfig_title=None, show_legend=True, save_path="tmp.png"):
    subplot_x = 1
    subplot_y = len(elements[0])
    print(f"[Note]subplot_x:{subplot_x}\t subplot_y:{subplot_y}")
    plt_init(figsize=(6.4 * subplot_y, 4 * subplot_x), subplot_flag=True)
    for sub_x in range(subplot_x):
        for sub_y in range(subplot_y):
            sub_fig_id = sub_x * subplot_y + sub_y + 1
            plot_x_list, plot_y_list = elements[sub_x][sub_y]
            print(f"[Note]plot_x_list:{plot_x_list}\t plot_y_list:{plot_y_list}")
            plt.subplot(subplot_x, subplot_y, sub_fig_id)
            # plt.clf()
            ax = plt.gca()
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=24,
                direction="in",
                bottom=True,
                top=True,
                left=True,
                right=True,
            )
            plot_simple_line(
                plot_x_list=elements[sub_x][sub_y][0],
                plot_y_list=elements[sub_x][sub_y][1],
                label_list=label_list,
                legend=(sub_fig_id == 2 and show_legend),
                xlabels=xlabels,
                ylabels=ylabels if sub_fig_id == 1 else None,
                # yticks=[0, 1, 2, 3, 4, 5],
            )
            # plt.title(f"({sub_x},{sub_y})")
            # show subfigure title on the bottom
            plt.title(subfig_title[sub_x][sub_y], y=-0.4, fontsize=24)
    # finalize and save
    if save_path is not None:
        plt_save_and_final(save_path=save_path)


def plot_stacked_bar(
    elements: np.ndarray,
    xticks,
    labels,
    width=1,
    xlabelticks=None,
    xlabels=None,
    ylabels=None,
    save_path="tmp.png",
):
    num_stacked_layers, num_bars = elements.shape

    plt_init()
    ax = plt.gca()
    bottom = np.zeros(num_bars)
    for yval, lab in zip(elements, labels):
        print(f"[Note]xticks:{xticks}\t yval:{yval}\t")
        ax.bar(xticks, yval, width=width, bottom=bottom, label=lab, edgecolor="k")
        bottom += yval
    font_size = 24
    if ylabels is not None:
        plt.ylabel(ylabels, fontsize=font_size)

    if xlabelticks is not None:
        plt.xticks(xlabelticks, xlabels, fontsize=font_size)
    # set ylim
    ymax = np.max(bottom)
    ax.set_ylim([0, ymax * 1.4])
    ax.legend(fontsize=font_size - 6, edgecolor="k", ncols=1)
    if save_path is not None:
        plt_save_and_final(save_path=save_path)


if __name__ == "__main__":
    raise NotImplementedError
    """
    test_elements = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1.5, 2.5, 3.5]]
    test_stack_elements = []
    for sub_element in test_elements:
        val = []
        for i in range(2):
            val.append([v / 2 for v in sub_element])
        test_stack_elements.append(val)

    test_labels = ["a", "b", "c", "d"]
    test_xlabel = [
        "x1",
        "x2",
        "x3",
    ]
    draw_figure_group_stacked_bar(
        elements=test_stack_elements,
        labels=test_labels,
        xlabel=test_xlabel,
        # ylabel="y",
        # yticks=[0, 2, 4, 6, 8, 10],
        save_path="tmp.png",
    )
    """


"""
def plt_bar(
    elements,
    labels,
    xlabels,
    ylabel=None,
    yticks=None,
    na_str="N/A",
    value_limit=8,
    save_path=None,
):
    plt_init()
    # fix parameter
    font_size = 14
    hatch_list = [None, "||", "--", "//", None]
    color_list = ["w", "w", "w", "w", "k"]
    ax = plt.gca()
    num_series = len(labels)
    num_elements_per_series = len(xlabels)
    offset = 5 - num_series
    offsets = [0 for _ in range(num_elements_per_series)]
    plot_x_list = []
    plot_y_list = []
    plot_label_list = []

    for i in range(num_series):
        plot_x = []
        # handle N/A
        plot_y = []
        plot_label = []
        for j, e in enumerate(elements[i]):
            if e == "":
                pass
            elif isfloat(e):
                plot_x.append(offsets[j])
                offsets[j] += 1
                val_e = float(e)
                if val_e < value_limit:
                    plot_y.append(float(e))
                    plot_label.append(round(float(e), 2))
                else:
                    plot_y.append(value_limit)
                    plot_label.append(round(float(e), 2))
            else:
                plot_x.append(offsets[j])
                offsets[j] += 1
                plot_y.append(0.01)
                plot_label.append(na_str)

        plot_x_list.append(plot_x)
        plot_y_list.append(plot_y)
        plot_label_list.append(plot_label)
    pre_offsets = [0] + list(itertools.accumulate(offsets))
    print(f"pre_offsets:{pre_offsets}")
    for i in range(num_series):
        plot_x = [pre_offsets[j] + 3 * j + plot_x_list[i][j] for j in range(len(plot_x_list[i]))]
        plot_y = plot_y_list[i]
        plot_label = plot_label_list[i]
        print(f"[Note]x:{plot_x}\t y:{plot_y}\t label:{plot_label}")
        container = ax.bar(
            plot_x,
            plot_y,
            width=1,
            edgecolor="k",
            hatch=hatch_list[offset + i],
            color=color_list[offset + i],
            label=labels[i],
        )
        ax.bar_label(container, plot_label, fontsize=5)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)

    if yticks is not None:
        ax.set_yticks(yticks, yticks, fontsize=font_size)

    plot_xticks = [(num_series + 3) * j + 0.5 * (num_series - 1) for j in range(num_elements_per_series)]

    ax.set_xticks(plot_xticks, xlabels, fontsize=font_size)
    ax.legend(fontsize=font_size, edgecolor="k", ncols=2)
    # plt.show()
    if save_path is not None:
        plt_save_and_final(save_path=save_path)


def isfloat(val):
    return all([[any([i.isnumeric(), i in [".", "e"]]) for i in val], len(val.split(".")) == 2])


def draw_figure(input_path="./data/data1.csv", tag="fig1", header_offset=1):
    with open(input_path, "r") as file:
        reader = csv.reader(file)
        headers = next(reader)[header_offset:]
        print(f"[Note]headers:{headers}")
        elements = []
        labels = []
        save_path = ""
        for row in reader:
            if len(row[0]) > 0:
                if len(elements) > 0:
                    plt_bar(elements, labels, headers, save_path=save_path)
                save_path = f"figures/{tag}_{row[0]}.pdf"
                elements.clear()
                labels.clear()

            labels.append(row[1])
            elements.append(row[2:])

            # print(f"[Note]len: {len(row)}")
        if len(elements) > 0:
            plt_bar(elements, labels, headers, save_path=save_path)


def count_nonempty_str(str_list):
    counter = 0
    for str in str_list:
        if len(str) > 0:
            counter += 1
    return counter


def draw_figure_all_in_one(
    input_path="./data/data2.csv",
    tag="fig2",
    header_offset=2,
    blank=3,
    value_limit=100,
    na_str="N/A",
):
    plt_init(figsize=(12, 4.8))
    ax = plt.gca()
    with open(input_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        major_labels = header[header_offset:]
        num_series = len(major_labels)
        print(f"[Note]headers:{major_labels}, num:{num_series}")
        elements = []
        minor_labels = []
        save_path = f"figures/{tag}.pdf"
        offsets = 0
        offsets_list = []
        ds_idx = []
        xticks = []
        xticks_minor = []
        xlbls = []
        for row_idx, row in enumerate(reader):
            if len(row[0]) > 0:
                ds_idx.append(row_idx)
            minor_labels.append(row[1])
            elements.append(row[2:])
            offsets_list.append(offsets)
            offsets += count_nonempty_str(elements[-1])

        print(f"[Note]ds_idx:{ds_idx}")
        print(f"[Note]minor_labels:{minor_labels}")
        # print(f"[Note]elements:{elements}")
        print(f"[Note]offsets_list:{offsets_list}")

        intra_offsets = [0 for _ in range(max(offsets_list))]
        num_rows = len(elements)

        font_size = 14
        hatch_list = [None, "||", "--", "//", None]
        color_list = ["r", "g", "b", "m", "k"]

        major_xticks_id = int(num_series / 2)
        for i in range(num_series):
            vals = [elements[row_id][i] for row_id in range(num_rows)]
            # print(f"[Note]{i}:{vals}")
            plot_x = []
            plot_y = []
            plot_y_label = []
            for j, val in enumerate(vals):
                if val == "":
                    continue
                plot_x.append(offsets_list[j] + j * blank + intra_offsets[j])
                intra_offsets[j] += 1
                if isfloat(val):
                    float_val = round(float(val), 2)
                    plot_y.append(min(value_limit, float_val))
                    plot_y_label.append(str(float_val))
                else:
                    plot_y.append(0.01)
                    plot_y_label.append(na_str)
            xticks_minor.extend(plot_x)
            if i == major_xticks_id:
                xticks = [plot_x[idx] for idx in ds_idx]
            print(f"[Note]plot_x:{plot_x}\t plot_y:{plot_y}\t label:{plot_y_label}")
            container = ax.bar(
                plot_x,
                plot_y,
                width=1,
                edgecolor="k",
                hatch=None,
                color=color_list[i],
                label=major_labels[i],
            )
            ax.bar_label(container, plot_y_label, fontsize=5)
        # ax.set_xticks(xticks)
        # ax.set_xticks(xticks_minor, minor=True)
        # ax.set_xticklabels(xlbls)
        plt.ylabel(header[0], fontsize=font_size)
        ax.legend(fontsize=font_size, edgecolor="k", ncols=2)
        print(f"[Note]xticks:{xticks}")
        print(f"[Note]xticks_minor:{xticks_minor}")
        print(f"[Note]xlabels:{xlbls}")
        plt_save_and_final(save_path=save_path)


# plot group stacked bar
def draw_figure_group_stacked_bar(
    elements,
    labels,
    xtickslabel=None,
    xlabel=None,
    ylabel=None,
    yticks=None,
    save_path="tmp.png",
):
    # init
    font_size = 14
    plt_init(figsize=(12, 4.8), labelsize=font_size)

    hatch_list = [None, "||", "--", "//", None, "\\\\"]
    color_list = ["r", "g", "b", "m", "k"]
    ax = plt.gca()
    num_series = len(labels)
    num_elements_per_series = len(elements[0][0])
    print(f"[Note]num_series:{num_series}\t num_elements_per_series:{num_elements_per_series}")
    offset = num_series + 1
    # plot
    stage_labels = ["Sampling", "Loading", "Training"]
    val_max = 0
    for series_id in range(num_series):
        plot_x = [offset * j + series_id for j in range(num_elements_per_series)]
        label = labels[series_id]
        bottom = [0 for _ in range(num_elements_per_series)]
        for j in range(len(elements[series_id])):
            plot_y = elements[series_id][j]
            print(f"[Note]plot_x:{plot_x}\t plot_y:{plot_y}\t label:{label}")
            container = ax.bar(
                plot_x,
                plot_y,
                bottom=bottom,
                width=1,
                edgecolor="k",
                hatch=hatch_list[series_id],
                # hatch=None,
                color=color_list[j],
                alpha=1 - (num_series - 1 - series_id) * 0.15,
                label=label + "-" + stage_labels[j],
            )
            bottom = [bottom[k] + plot_y[k] for k in range(num_elements_per_series)]
            val_max = max(val_max, max(bottom))
    print(f"[Note]val_max:{val_max}")
    if xtickslabel is not None:
        xticks = [1.5 + offset * j for j in range(num_elements_per_series)]
        ax.set_xticks(xticks, xtickslabel, fontsize=font_size)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=font_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)

    if yticks is not None:
        ax.set_yticks(yticks, yticks, fontsize=font_size)
    else:
        ax.set_ylim([0, val_max * 1.4])
    # legend
    ax.legend(
        fontsize=font_size,
        edgecolor="k",
        ncols=num_series,
        handlelength=1.0,
        columnspacing=1.0,
    )
    # finalize and save
    if save_path is not None:
        plt_save_and_final(save_path=save_path)

"""
