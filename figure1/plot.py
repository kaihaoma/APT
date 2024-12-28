import csv
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 1
FONT_SIZE = 13
COLOR_LIST = ["white", "silver", "bisque", "skyblue"]
HATCH_LIST = ["||", None, "//"]

input_dim_path = "outputs/input_dim"
hidden_dim_path = "outputs/hidden_dim"


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


def draw_bar(
    path_dir: str,
    draw_dataset: list[str] = None,
    xlabels=[8, 32, 128, 512],
    xlabels_fontsize=FONT_SIZE,
    xlabel="Hidden Dimension",
    yticks=[0, 5, 10, 15, 20, 25],
    ylimit=None,
    legend=True,
    save_path="tmp.pdf",
):
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
    assert len(csv_path_list) == 1
    num_subfigures = len(csv_path_list)
    fig, ax = plt.subplots(1, num_subfigures)
    fig.set_size_inches(num_subfigures * 6, 2)
    width = 1.0
    width_interval = 1.0
    num_systems = 4
    num_elements_per_system = len(xlabels)
    list_system = ["GDP", "NFP", "SNP", "DNP"]

    for cp in csv_path_list:
        full_path = os.path.join(path_dir, cp)
        dataset_name = get_dataset_name_by_path_name(cp)
        print(f"[Note] Full path: {full_path}\t dataset_name:{dataset_name}")
        header, elements = read_csv(full_path)
        e0 = len(elements)
        e1 = len(elements[0])
        print(f"[Note] elements shape :{e0} * {e1}")

        ax.set_yticks(yticks)
        ylim = yticks[-1] if ylimit is None else ylimit
        ax.set_ylim(0, ylim)
        ax.tick_params(axis="y", labelsize=10)  # 设置 y 轴刻度标签的字体大小

        ax.set_xticks([])
        # ax.set_xticklabels()

        # ax.set_xlabel(dataset_name, fontsize=font_size)
        # ax.set_title(dataset_name, fontsize=font_size, y=-0.35)
        ax.grid(axis="y", linestyle="--")

        for system_id in range(num_systems):
            plot_x = [
                (num_systems * width + width_interval) * j + system_id * width
                for j in range(num_elements_per_system)
            ]
            sys_elements = np.array(
                [
                    float(elements[num_systems * j + system_id][-1])
                    for j in range(num_elements_per_system)
                ]
            )
            plot_elements = np.minimum(ylim, sys_elements)
            container = ax.bar(
                plot_x,
                plot_elements,
                width=width,
                edgecolor="k",
                color=COLOR_LIST[system_id],
                label=list_system[system_id],
                zorder=10,
                # alpha=0.9,
            )

            bar_elements = np.round(sys_elements, decimals=1).tolist()
            for it, ele in enumerate(bar_elements):
                if ele == 37.4:
                    bar_elements[it] = str(ele) + "   "
                elif ele == 20.7:
                    bar_elements[it] = "   " + str(ele)
            ax.bar_label(
                container,
                bar_elements,
                fontsize=11,
                zorder=20,
                # fontweight="bold",
            )

    width_per_elements = width * num_systems + width_interval
    plot_xticks = [
        width_per_elements * j + (width / 2) * (num_systems - 1)
        for j in range(num_elements_per_system)
    ]
    ax.set_xlim(-width, plot_x[-1] + width)
    ax.set_xticks(plot_xticks, xlabels, fontsize=xlabels_fontsize)
    ax.set_xlabel(f"{xlabel} ({dataset_name})", fontsize=font_size)

    # legend
    if legend:
        ax.legend(
            bbox_to_anchor=(0.91, 1.03),
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
    ax.set_ylabel("Epoch Time (s)", fontsize=font_size)
    ax.set_yticklabels(yticks, fontsize=font_size)

    # save the figure
    plt_save_and_final(save_path)


def plt_save_and_final(save_path):
    print(f"[Note]Save to {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    # [NOTE] Introduction Figure

    # draw input_dim
    preprocess_data(
        file_path="outputs/results_input_dim.csv",
        include_dict={
            "world_size": 8,
            "gpu_cache_mem": 4,
            "model": "SAGE",
            "fanout": "[10, 10, 10]",
        },
        key_name="input_dim",
        save_dir=input_dim_path,
        xlabels=[64, 128, 256],
    )
    # (1) input_dim-papers
    draw_bar(
        path_dir=input_dim_path,
        draw_dataset=["papers"],
        xlabels=[64, 128, 256],
        xlabel="Input Dimension",
        yticks=[0, 3, 6, 9, 12],
        ylimit=13,
        legend=True,
        save_path="outputs/input_dim.pdf",
    )

    # draw hidden_dim
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
    draw_bar(
        path_dir=hidden_dim_path,
        draw_dataset=["friendster"],
        xlabels=[8, 32, 128, 512],
        xlabel="Hidden Dimension",
        yticks=[0, 3, 6, 9, 12, 15],
        legend=False,
        save_path="outputs/hidden_dim.pdf",
    )
