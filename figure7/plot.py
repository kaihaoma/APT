import csv
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1
FONT_SIZE = 13
COLOR_LIST = ["white", "silver", "bisque", "skyblue"]
HATCH_LIST = ["||", None, "//"]

hidden_dim_path = "./outputs/hidden_dim"
fanout_path = "./outputs/fanout"
gpu_cache_path = "./outputs/gpu_cache_mem"


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


def get_dataset_name_by_path_name(path_name):
    if "papers" in path_name:
        return "Papers"
    elif "friendster" in path_name:
        return "Friendster"
    elif "igbfull" in path_name:
        return "IGB260M"
    else:
        raise ValueError


def rerange_by_key(input, key_list=["papers", "friendster", "igbfull"]):
    output = []
    for key in key_list:
        for element in input:
            if key in element:
                output.append(element)
                break
    return output


def sys_to_id(sys):
    if sys == "DP":
        return 0
    elif sys == "MP":
        return 1
    elif sys == "SP":
        return 2
    elif sys == "NP":
        return 3
    else:
        raise NotImplementedError


def decide_include_dict(key_dict, include_dict):
    for key, val in include_dict.items():
        if key not in key_dict:
            return False
        if key_dict[key] != val:
            return False
    return True


def get_xid(xlabels, key_dict, include_dict, key_name):
    if not decide_include_dict(key_dict, include_dict):
        return None
    xval = key_dict[key_name]
    xid = None
    select_xlabels = (
        xlabels if not isinstance(xlabels, dict) else xlabels[key_dict["graph"]]
    )
    for it, xl in enumerate(select_xlabels):
        if xval == xl:
            xid = it
            break
    return xid


def preprocess_data(
    file_path,
    key_name="hidden_dim",
    include_dict={},
    save_dir=None,
    xlabels=None,
):
    labels = ["GDP", "NFP", "SNP", "DNP"]
    lines = []
    if isinstance(file_path, str):
        file_path = [file_path]
    for fp in file_path:
        header, line = read_csv(fp, has_header=None)
        lines.extend(line)
    num_elements_per_series = len(xlabels)
    num_series = len(labels)
    vals_dict = {}
    records_dict = {}
    for line in lines:
        # decode tag
        tag_list = line[0].split("_")
        graph = tag_list[0]
        world_size = int(tag_list[1][1:])
        method = tag_list[2]
        offset = 1 if "variance" in line[0] else 0
        sys = tag_list[3 + offset]
        model = tag_list[4 + offset]
        gpu_cache_mem = int(tag_list[6 + offset][2:-2])
        if "of16" in tag_list[5 + offset]:
            nl = float(tag_list[5 + offset][2:-4])

        input_dim = int(line[4])
        hidden_dim = int(line[5])
        sample = float(line[-4])
        load = float(line[-3])
        train = float(line[-2])
        fanout = line[7]

        key_dict = {
            "graph": graph,
            "world_size": world_size,
            "method": method,
            "sys": sys,
            "model": model,
            "gpu_cache_mem": gpu_cache_mem,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "fanout": fanout,
        }

        # print(f"[Note] key_dict:{key_dict}\t include_dict:{include_dict}")
        xid = get_xid(
            xlabels=xlabels,
            key_dict=key_dict,
            include_dict=include_dict,
            key_name=key_name,
        )
        if xid is None:
            continue
        # print(f"[Note]key_dict:{key_dict}, xid:{xid}")
        if key_name != "hidden_dim":
            select_key = f"{graph}_w{world_size}_{model}_hidden_dim{hidden_dim}"
        else:
            select_key = f"{graph}_w{world_size}_{model}"
        if select_key not in vals_dict:
            vals_dict[select_key] = [
                [np.zeros((3, num_elements_per_series)) for i in range(num_series)],
                [np.zeros((3, num_elements_per_series)) for i in range(num_series)],
            ]

        yvals = vals_dict[select_key][0]
        bottoms = vals_dict[select_key][1]
        sys_id = sys_to_id(sys)

        yvals[sys_id][0][xid] = sample / 1000
        yvals[sys_id][1][xid] = load / 1000
        yvals[sys_id][2][xid] = train / 1000

        # record multi_records
        record_key = f"{select_key}_{sys}_{key_dict[key_name]}"
        record_val = (sample + load + train) / 1000
        if record_key not in records_dict:
            records_dict[record_key] = []
        records_dict[record_key].append(record_val)

    for key, vals in vals_dict.items():
        yvals = vals[0]
        bottoms = vals[1]
        for i in range(num_series):
            for j in range(num_elements_per_series):
                for k in range(2):
                    bottoms[i][k + 1][j] = bottoms[i][k][j] + yvals[i][k][j]

    print(f"[Note] processed csv save_dir:{save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    for subplot_id, (key, vals) in enumerate(vals_dict.items()):
        to_subplot_id = 0
        if "papers" in key:
            to_subplot_id = 0
        elif "friendster" in key:
            to_subplot_id = 1
        elif "igbfull" in key:
            to_subplot_id = 2
        print(f"[Note] subplot:{to_subplot_id} \t key:{key}")
        yvals = vals[0]
        bottoms = vals[1]
        graph_name = key.split("_")[0]
        select_xlabels = (
            xlabels if not isinstance(xlabels, dict) else xlabels[graph_name]
        )
        if key_name == "fanout":
            select_xlabels = ["[10, 5]", "[15, 10]", "[10, 10, 10]", "[20, 15, 10]"]
        print(f"[Note] key:{key} select_xlabels:{select_xlabels}")
        print(f"[Note] graph_name:{graph_name}")
        # save to csv

        csv_path = os.path.join(save_dir, f"{key}.csv")
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            headers = [
                "Dataset",
                key,
                "System",
                "Sampling",
                "FeatLoading",
                "Training",
                "Total",
            ]
            writer.writerow(headers)
            print(f"[Note] headers:{headers}")
            print(f"[Note] write {num_series} * {num_elements_per_series}")
            for i in range(num_elements_per_series):
                for j in range(num_series):
                    writer.writerow(
                        [
                            key.split("_")[0],
                            select_xlabels[i],
                            labels[j],
                            yvals[j][0][i],
                            yvals[j][1][i],
                            yvals[j][2][i],
                            yvals[j][0][i] + yvals[j][1][i] + yvals[j][2][i],
                        ]
                    )


def draw_stacked_bar(
    path_dir: str,
    draw_dataset: list[str] = None,
    xlabels=[8, 32, 128, 512],
    xlabels_fontsize=FONT_SIZE,
    xlabel="Hidden Dimension",
    yticks=[0, 5, 10, 15, 20, 25],
    ylimit=None,
    scatter=None,
    legend=True,
    scatter_label="APT",
    save_path="tmp.pdf",
) -> None:
    def filter_dataset(path_list, filter_list):
        if filter_list is None:
            return path_list
        output = []
        for path in path_list:
            for ft in filter_list:
                if ft in path:
                    output.append(path)
                    break
        return output

    # set font size
    font_size = FONT_SIZE
    csv_path_list = filter_dataset(rerange_by_key(os.listdir(path_dir)), draw_dataset)
    num_subfigures = len(csv_path_list)
    print(f"[Note] #csv: {num_subfigures}\t list:{csv_path_list}")
    fig, axes = plt.subplots(1, num_subfigures)
    if num_subfigures == 1:
        axes = [axes]
    fig.set_size_inches(num_subfigures * 5, 2)
    plt.subplots_adjust(wspace=0, hspace=0)

    yticks_limit = yticks[-1] if ylimit is None else ylimit
    width = 1.0
    width_interval = 1.0
    num_systems = 4
    num_elements_per_system = len(xlabels)
    num_stages = 3
    list_stage = ["Sampling", "FeatLoading", "Training"]
    list_system = ["GDP", "NFP", "SNP", "DNP"]
    sample_offset = -4

    for sub_fig_id, cp in enumerate(csv_path_list):
        full_path = os.path.join(path_dir, cp)
        dataset_name = get_dataset_name_by_path_name(cp)
        print(f"[Note] Path{sub_fig_id}: {full_path}\t dataset_name:{dataset_name}")
        header, elements = read_csv(full_path)
        e0 = len(elements)
        e1 = len(elements[0])
        print(f"[Note] elements shape :{e0} * {e1}")
        # [NOTE] header:['Dataset', 'papers_w8_SAGE', 'System', 'Sampling', 'FeatLoading', 'Training', 'Total']

        ax = axes[sub_fig_id]
        ax.set_yticks(yticks)
        # ax.set_ylim(0, yticks[-1] * 1.05)
        ax.set_ylim(0, yticks_limit)
        ax.tick_params(axis="y", labelsize=10)

        ax.set_xticks([])
        # ax.set_xticklabels()

        # ax.set_xlabel(dataset_name, fontsize=font_size)
        # ax.set_title(dataset_name, fontsize=font_size, y=-0.35)
        ax.grid(axis="y", linestyle="--")

        # plot bar per system and then per stack bar
        bar_labels_list = []
        # [NOTE] for scatter
        min_val_x = np.zeros(num_systems)
        min_val_y = np.zeros(num_systems)

        for system_id in range(num_systems):
            system_elements = [
                np.array(
                    [
                        float(
                            elements[num_systems * j + system_id][
                                stage_id2 + sample_offset
                            ]
                        )
                        for j in range(num_elements_per_system)
                    ]
                )
                for stage_id2 in range(num_stages)
            ]
            system_sum = sum(system_elements)
            round_bar_label = np.round(system_sum, decimals=1)
            bar_labels_list.append(round_bar_label)

        for system_id in range(num_systems):
            plot_x = [
                (num_systems * width + width_interval) * j + system_id * width
                for j in range(num_elements_per_system)
            ]
            # print(f"[Note] system#{system_id}: plot_x:{plot_x}")
            bottom = np.zeros(num_elements_per_system)
            system_elements = [
                np.array(
                    [
                        float(
                            elements[num_systems * j + system_id][
                                stage_id2 + sample_offset
                            ]
                        )
                        for j in range(num_elements_per_system)
                    ]
                )
                for stage_id2 in range(num_stages)
            ]

            # print(f"[Note] system_elements:{system_elements}")
            # [NOTE] scale down elements to fit yticks limit
            system_sum = sum(system_elements)
            system_ratio = np.maximum(system_sum / yticks_limit, 1.0)
            round_bar_label = np.round(system_sum, decimals=1)

            # [NOTE] for scatter
            for ele_id in range(num_elements_per_system):
                if scatter is not None and scatter[sub_fig_id][ele_id] == system_id:
                    min_val_x[ele_id] = plot_x[ele_id]
                    min_val_y[ele_id] = system_sum[ele_id] + 0.06 * yticks_limit

            for i in range(num_stages):
                for j in range(num_elements_per_system):
                    system_elements[i][j] /= system_ratio[j]

            for stage_id in range(num_stages):
                stage_elements = system_elements[stage_id]
                container = ax.bar(
                    plot_x,
                    stage_elements,
                    bottom=bottom,
                    width=width,
                    edgecolor="k",
                    hatch=HATCH_LIST[stage_id],
                    color=COLOR_LIST[system_id],
                    label=list_system[system_id] if stage_id == 1 else None,
                    zorder=10,
                    # alpha=0.9,
                )
                bottom += stage_elements

            if sub_fig_id == 0 and system_id == num_systems - 1:
                for stage_id in range(num_stages):
                    ax.bar(
                        plot_x,
                        np.zeros_like(stage_elements),
                        bottom=np.zeros_like(stage_elements),
                        width=width,
                        edgecolor="k",
                        hatch=HATCH_LIST[stage_id],
                        color="none",
                        label=list_stage[stage_id],
                    )

            plot_bar_label = []

            for i in range(num_elements_per_system):
                if round_bar_label[i] > yticks_limit:
                    if round_bar_label[i] == 111.2:
                        plot_str = "111"
                    else:
                        plot_str = str(round_bar_label[i])
                    if (
                        system_id - 1 >= 0
                        and bar_labels_list[system_id - 1][i] > yticks_limit
                    ):
                        plot_str = " " * 4 + plot_str
                    if (
                        system_id + 1 < num_systems
                        and bar_labels_list[system_id + 1][i] > yticks_limit
                    ):
                        plot_str = plot_str + " " * 4
                elif round_bar_label[i] < 0.1:
                    plot_str = "OM"
                else:
                    plot_str = ""
                plot_bar_label.append(plot_str)

            ax.bar_label(
                container,
                plot_bar_label,
                fontsize=11,
                zorder=20,
                # fontweight="bold",
            )
            # bar_labels_list.append(plot_bar_label)
            width_per_elements = width * num_systems + width_interval
            plot_xticks = [
                width_per_elements * j + (width / 2) * (num_systems - 1)
                for j in range(num_elements_per_system)
            ]
            ax.set_xlim(-width, plot_x[-1] + width)
            ax.set_xticks(plot_xticks, xlabels, fontsize=xlabels_fontsize)
            ax.set_xlabel(f"{xlabel} ({dataset_name})", fontsize=font_size)

        if scatter:
            ax.scatter(
                min_val_x,
                min_val_y,
                color="r",
                label=scatter_label,
                zorder=20,
                marker="*",
                s=200,
                edgecolors="k",
            )

    # legend
    if legend:
        left_pos = 2.5 if scatter is None else 2.65
        axes[0].legend(
            bbox_to_anchor=(left_pos, 1.08),
            ncol=8,
            loc="lower right",
            # fontsize=font_size,
            # markerscale=3,
            labelspacing=0.2,
            edgecolor="black",
            facecolor="white",
            framealpha=1,
            shadow=False,
            # fancybox=False,
            handlelength=2,
            handletextpad=0.5,
            columnspacing=0.5,
            prop={"size": font_size},
        ).set_zorder(100)
    # set labels and ticks
    axes[0].set_ylabel("Epoch Time (s)", fontsize=font_size)
    axes[0].set_yticklabels(yticks, fontsize=font_size)
    for i in range(1, num_subfigures):
        axes[i].set_yticklabels([])

    # save the figure
    plt_save_and_final(save_path)


def plt_save_and_final(save_path):
    print(f"[Note]Save to {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    # [NOTE] Main Exp
    # draw hidden_dim
    scatter_hidden_dim = [[3, 3, 0, 0], [2, 2, 3, 0], [3, 3, 3, 0]]
    preprocess_data(
        file_path="outputs/results_hidden_dim.csv",
        include_dict={
            "world_size": 8,
            "gpu_cache_mem": 4,
            "model": "SAGE",
            "fanout": "[10, 10, 10]",
        },
        key_name="hidden_dim",
        save_dir=hidden_dim_path,
        xlabels=[8, 32, 128, 512],
    )
    draw_stacked_bar(
        path_dir=hidden_dim_path,
        xlabels=[8, 32, 128, 512],
        xlabel="Hidden Dimension",
        yticks=[0, 3, 6, 9, 12, 15],
        ylimit=16,
        scatter=scatter_hidden_dim,
        legend=True,
        save_path="outputs/hidden_dim.pdf",
    )

    # draw fanout
    scatter_fanout = [[0, 0, 0, 0], [0, 2, 2, 2], [0, 3, 2, 2]]
    preprocess_data(
        file_path="outputs/results_fanout.csv",
        include_dict={
            "world_size": 8,
            "gpu_cache_mem": 4,
            "model": "SAGE",
            "hidden_dim": 32,
        },
        key_name="fanout",
        save_dir=fanout_path,
        xlabels=["[5, 10]", "[10, 15]", "[10, 10, 10]", "[10, 15, 20]"],
    )
    draw_stacked_bar(
        path_dir=fanout_path,
        xlabels=["10,5", "15,10", "10,10,10", "20,15,10"],
        xlabel="Fanout",
        yticks=[0, 2, 4, 6, 8, 10],
        scatter=scatter_fanout,
        legend=False,
        save_path="outputs/fanout.pdf",
    )

    # draw vary gpu cache memory
    scatter_gpu_cache_mem = [[0, 3, 3, 0], [0, 2, 2, 2], [0, 2, 2, 3]]
    preprocess_data(
        file_path="outputs/results_cache_mem.csv",
        include_dict={
            "hidden_dim": 32,
            "world_size": 8,
            "model": "SAGE",
            "fanout": "[10, 10, 10]",
        },
        key_name="gpu_cache_mem",
        save_dir=gpu_cache_path,
        xlabels=[0, 2, 4, 6],
    )
    draw_stacked_bar(
        path_dir=gpu_cache_path,
        xlabels=[0, 2, 4, 6],
        xlabel="GPU Cache Memory",
        yticks=[0, 3, 6, 9, 12],
        scatter=scatter_gpu_cache_mem,
        legend=False,
        save_path="outputs/gpu_cache_mem.pdf",
    )
