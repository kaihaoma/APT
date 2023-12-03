import numpy as np
import itertools
import csv
import os
import draw_utils


# tag example
# friendster_variance[2]_t0_SP_GAT_nl-1of8_cm3GB
def filter_list(x_list, y_list):
    assert len(x_list) == len(y_list)
    filter_x_list = []
    filter_y_list = []
    for x, y in zip(x_list, y_list):
        if y > 0:
            filter_x_list.append(x)
            filter_y_list.append(y)
    return filter_x_list, filter_y_list


# Single-machine: fixed nl, varied gpu cache mem
# Multi-machine: fixed nl, varied gpu cache mem
#              or fixed gpu cache mem, varied nl


def draw_multimachines_mainexp(path, output_prefix="./outputs/figures/mult-machine/"):
    # pre-defined
    labels_list = ["DP", "NP", "SP", "MP", "DP+NP", "DP+SP"]
    n_labels = len(labels_list)

    def labels_to_idx(sys):
        if sys == "DP":
            return 0
        elif sys == "NP":
            return 1
        elif sys == "SP":
            return 2
        elif sys == "MP":
            return 3
        elif sys == "DP+NP":
            return 4
        elif sys == "DP+SP":
            return 5
        else:
            raise ValueError(f"[Error] sys:{sys}")

    headers, elements = draw_utils.read_csv(path, has_header=None)
    plot_x_list = [[] for _ in range(n_labels)]
    plot_y_list = [[] for _ in range(n_labels)]

    for lid, element in enumerate(elements):
        if element[0].startswith("unused"):
            continue
        # decode tag
        tag_list = element[0].split("_")
        graph = tag_list[0]
        method = tag_list[2]
        offset = 1 if "variance" in element[0] else 0
        sys = tag_list[4 + offset]
        model = tag_list[5 + offset]
        gpu_cache_mem = int(tag_list[7 + offset][2:-2])
        if "of16" in tag_list[6 + offset]:
            nl = float(tag_list[6 + offset][2:-4])

        epoch_time = float(element[-1]) / 1000.0

        x_val = nl
        y_val = epoch_time
        print(
            f"[Note] graph:{graph}\t method:{method}\t sys:{sys}\t model:{model}\t gpu_cache_mem:{gpu_cache_mem}\t nl:{nl}\t xval:{x_val}\t yval:{y_val}"
        )

        sys_id = labels_to_idx(sys)
        plot_x_list[sys_id].append(x_val)
        plot_y_list[sys_id].append(y_val)

    output_name = os.path.basename(path).split(".")[0]
    print(f"[Note]output_name:{output_name}")
    # broadcast SP and MP
    for i in range(n_labels):
        if len(plot_x_list[i]) == 1:
            plot_x_list[i] = [val for val in plot_x_list[i - 1]]

            plot_y_list[i] = [plot_y_list[i][0]] * len(plot_y_list[i - 1])
        print(f"[Note]{labels_list[i]}: plot_x_list:{plot_x_list[i]}\t plot_y_list:{plot_y_list[i]}")

    # sort x ticks
    y_max = 0
    for i in range(n_labels):
        x_val, y_val = plot_x_list[i], plot_y_list[i]
        idx = sorted(range(len(x_val)), key=x_val.__getitem__)
        plot_x_list[i] = [x_val[i] for i in idx]
        plot_y_list[i] = [y_val[i] for i in idx]
        y_max = max(np.max(y_val), y_max)

    os.makedirs(output_prefix, exist_ok=True)
    draw_utils.plot_line(
        plot_x_list=plot_x_list,
        plot_y_list=plot_y_list,
        labels=labels_list,
        # xticks=list(range(9)),
        # yticks=list(range(0, math.ceil(y_max) + 5, 5)),
        # xlabels="GPU Cache Memory (GB)" if fixed == "nl" else "Number of Local Node Features",
        xlabels="CPU Memory (GB)",
        ylabels="Epoch Time (s)",
        legends_font_size=18,
        save_path=os.path.join(output_prefix, f"{output_name}.png"),
    )


def draw_mainexp(path_list=["./logs/ap/Nov15_single_machine_stable.csv"], fixed="nl", output_prefix="./outputs/figures/single-machine/"):
    assert fixed in ["nl", "gpu_cache_mem"]
    varied = "gpu_cache_mem" if fixed == "nl" else "nl"
    data_dicts = {}
    fig_key_set = set()
    sys_set = set()

    for path in path_list:
        headers, elements = draw_utils.read_csv(path, has_header=None)

        for lid, element in enumerate(elements):
            # decode tag
            tag_list = element[0].split("_")
            for i in range(len(tag_list)):
                print(f"[Note]tag_list[{i}]:{tag_list[i]}")
            graph = tag_list[0]
            method = tag_list[2]
            offset = 1 if "variance" in element[0] else 0
            sys = tag_list[4 + offset]
            model = tag_list[5 + offset]
            gpu_cache_mem = int(tag_list[7 + offset][2:-2])
            if "of16" in tag_list[6 + offset]:
                nl = int(tag_list[6 + offset][2:-4])
            else:
                nl = int(tag_list[6 + offset][2:-3])

            epoch_time = float(element[-1]) / 1000.0
            # (gpu_cache_mem, epoch_time)
            key = (graph, model, method, gpu_cache_mem if fixed == "gpu_cache_mem" else nl, sys)
            fig_key = (graph, model, method, gpu_cache_mem if fixed == "gpu_cache_mem" else nl)
            fig_key_set.add(fig_key)
            sys_set.add(sys)

            x_val = gpu_cache_mem if fixed == "nl" else nl
            y_val = epoch_time
            if key not in data_dicts:
                data_dicts[key] = [[], []]

            print(f"[Note] Insert key:{key} values:{x_val}, {y_val}")
            while len(data_dicts[key][0]) <= x_val:
                data_dicts[key][0].append(len(data_dicts[key][0]))
                data_dicts[key][1].append(-1)

            data_dicts[key][0][x_val] = x_val
            data_dicts[key][1][x_val] = y_val

    key_lists = list(data_dicts.keys())
    for key in key_lists:
        print(f"[Note]data dict key: {key}")

    # dataset_list = list(dataset_dicts)
    label_list = ["DP", "NP", "SP", "MP"]
    # one dataset, model a graph
    for fig_key in fig_key_set:
        print(f"[Note]fig_key:{fig_key}")
        plot_x_list = []
        plot_y_list = []
        for sys in label_list:
            key = (*fig_key, sys)
            if key not in data_dicts:
                continue
            x_list = data_dicts[key][0]
            y_list = data_dicts[key][1]
            filter_x_list, filter_y_list = filter_list(x_list, y_list)
            print(f"[Note]key:{key}\t data_dicts:{filter_x_list}\t filter_y_list:{filter_y_list}")
            plot_x_list.append(filter_x_list)
            plot_y_list.append(filter_y_list)
        os.makedirs(output_prefix, exist_ok=True)
        draw_utils.plot_line(
            plot_x_list=plot_x_list,
            plot_y_list=plot_y_list,
            labels=label_list,
            # xticks=list(range(9)),
            # yticks=[0, 2, 4, 6],
            xlabels="GPU Cache Memory (GB)" if fixed == "nl" else "Number of Local Node Features",
            ylabels="Epoch Time (s)",
            legends_font_size=18,
            save_path=os.path.join(output_prefix, f"{fig_key[0]}_{fig_key[1]}_{fig_key[2]}_{fig_key[3]}.png"),
        )


def draw_accuracy(path_prefix="../outputs/accuracy", filter_model="SAGE", filer_dataset="papers", filer_worldsize=4):
    plot_x_list = []
    plot_y_list = []
    labels = []
    for file in sorted(os.listdir(path_prefix)):
        file_split = file.split("_")
        model, sys, dataset, num_workers = file_split
        print(f"[Note]model:{model}\t sys:{sys}\t dataset:{dataset}\t num_workers:{num_workers}")
        # filter by model, dataset, world_size
        if filter_model not in model or filer_dataset not in dataset or str(filer_worldsize) not in num_workers:
            continue
        path = os.path.join(path_prefix, file)
        print(f"[Note] filter model:{model}\t sys:{sys}\t dataset:{dataset}\t num_workers:{num_workers}")
        # read txt
        plot_x = []
        plot_y = []
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                line_split = line.split(" ")
                plot_y.append(float(line_split[7]))

        plot_x = list(range(len(plot_y)))
        print(f"[Note]plot_x:{plot_x}\t plot_y:{plot_y}")
        plot_x_list.append(plot_x)
        plot_y_list.append(plot_y)
        labels.append(sys)
        if len(plot_x_list) == 5:
            draw_utils.plot_line(
                plot_x_list=plot_x_list,
                plot_y_list=plot_y_list,
                labels=labels,
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                xlabels="Numbe of epochs",
                ylabels="Accuracy",
                save_path=f"../outputs/figures/accuracy/{model}_{dataset}_{num_workers}.png",
            )
            plot_x_list = []
            plot_y_list = []
            labels = []


def draw_micro(
    path_list=["./logs/ap/Nov15_single_machine_stable.csv"], fixed="nl", output_prefix="./outputs/figures/single-machine/", additional_key=None
):
    assert fixed in ["nl", "gpu_cache_mem"]
    varied = "gpu_cache_mem" if fixed == "nl" else "nl"
    data_dicts = {}
    fig_key_set = set()
    sys_set = set()

    if additional_key == "hidden_dim":
        additional_key = 5
    elif additional_key == "input_dim":
        additional_key = 4
    else:
        raise KeyError

    for path in path_list:
        headers, elements = draw_utils.read_csv(path, has_header=None)

        for lid, element in enumerate(elements):
            if element[0].startswith("unused"):
                continue
            # decode tag
            tag_list = element[0].split("_")
            for i in range(len(tag_list)):
                print(f"[Note]tag_list[{i}]:{tag_list[i]}")
            graph = tag_list[0]
            method = tag_list[2]
            offset = 1 if "variance" in element[0] else 0
            sys = tag_list[4 + offset]
            model = tag_list[5 + offset]
            gpu_cache_mem = int(tag_list[7 + offset][2:-2])
            if "of16" in tag_list[6 + offset]:
                nl = int(tag_list[6 + offset][2:-4])
            else:
                nl = int(tag_list[6 + offset][2:-3])

            aux_key = int(element[additional_key])
            epoch_time = float(element[-1]) / 1000.0
            # (gpu_cache_mem, epoch_time)
            key = (graph, model, method, gpu_cache_mem if fixed == "gpu_cache_mem" else nl, aux_key, sys)
            fig_key = (graph, model, method, gpu_cache_mem if fixed == "gpu_cache_mem" else nl, aux_key)
            fig_key_set.add(fig_key)
            sys_set.add(sys)

            x_val = gpu_cache_mem if fixed == "nl" else nl
            y_val = epoch_time
            if key not in data_dicts:
                data_dicts[key] = [[], []]

            print(f"[Note] Insert key:{key} values:{x_val}, {y_val}")
            while len(data_dicts[key][0]) <= x_val:
                data_dicts[key][0].append(len(data_dicts[key][0]))
                data_dicts[key][1].append(-1)

            data_dicts[key][0][x_val] = x_val
            data_dicts[key][1][x_val] = y_val

    key_lists = list(data_dicts.keys())
    for key in key_lists:
        print(f"[Note]data dict key: {key}")

    # dataset_list = list(dataset_dicts)
    label_list = ["DP", "NP", "SP", "MP"]
    # one dataset, model a graph
    for fig_key in fig_key_set:
        print(f"[Note]fig_key:{fig_key}")
        plot_x_list = []
        plot_y_list = []
        for sys in label_list:
            key = (*fig_key, sys)
            if key not in data_dicts:
                continue
            x_list = data_dicts[key][0]
            y_list = data_dicts[key][1]
            filter_x_list, filter_y_list = filter_list(x_list, y_list)
            print(f"[Note]key:{key}\t data_dicts:{filter_x_list}\t filter_y_list:{filter_y_list}")
            plot_x_list.append(filter_x_list)
            plot_y_list.append(filter_y_list)
        os.makedirs(output_prefix, exist_ok=True)
        draw_utils.plot_line(
            plot_x_list=plot_x_list,
            plot_y_list=plot_y_list,
            labels=label_list,
            # xticks=list(range(9)),
            # yticks=[0, 2, 4, 6],
            xlabels="GPU Cache Memory (GB)" if fixed == "nl" else "Number of Local Node Features",
            ylabels="Epoch Time (s)",
            legends_font_size=18,
            save_path=os.path.join(output_prefix, f"{fig_key[0]}_{fig_key[1]}_{fig_key[2]}_{fig_key[3]}_{fig_key[4]}.png"),
        )


if __name__ == "__main__":
    fixed_list = ["nl", "gpu_cache_mem"]

    # multi-machines
    multi_machine_path_prefix = "../outputs/speed/multi_machine/"
    multi_machine_files = os.listdir(multi_machine_path_prefix)
    path_list = []
    for file in multi_machine_files:
        if "_10_10_10" in file:
            path_list.append(os.path.join(multi_machine_path_prefix, file))
    print(f"[Note]path_list:{path_list}")
    for path in path_list:
        draw_multimachines_mainexp(path=path, output_prefix="../outputs/figures/multi_machine")

    # single machine
    # single_machine_path_list = []
    # single_machine_path_list.append("../outputs/speed/single_machine/ALL_papers.csv")
    # single_machine_path_list.append("../logs/ap/papers_w4_metis_all.csv")
    # draw_mainexp(path_list=single_machine_path_list, fixed="nl", output_prefix="../outputs/figures/single_machine")

    # micro on hidden dim
    # single_machine_path_list = []
    # single_machine_path_list.append("../outputs/micro/varying_hidden_dim.csv")
    # draw_micro(path_list=single_machine_path_list, fixed="nl", output_prefix="../outputs/figures/micro/hidden_dim", additional_key="hidden_dim")

    # micro on input dim
    # single_machine_path_list = []
    # single_machine_path_list.append("../outputs/micro/varying_input_dim.csv")
    # draw_micro(path_list=single_machine_path_list, fixed="nl", output_prefix="../outputs/figures/micro/input_dim", additional_key="input_dim")

    # draw_accuracy()
