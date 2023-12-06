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
def labels_to_idx(sys):
    if sys == "DP":
        return 0
    elif sys == "MP":
        return 1
    elif sys == "SP":
        return 2
    elif sys == "NP":
        return 3
    else:
        return 4


ori_label_list = ["DP", "MP", "SP", "NP"]
map_label_list = ["GDP", "NFP", "SNP", "NFP"]


def draw_multimachines_mainexp(path_list, output_prefix="./outputs/figures/mult-machine/"):
    # pre-defined
    # labels_list = ["DP", "NP", "SP", "MP", "DP+NP", "DP+SP"]
    # labels_list = ["DP", "NP", "SP", "MP"]
    n_labels = 4
    """
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
    """

    plot_values = []
    subfig_title = [["PS", "FS", "IH"]]
    for i in range(1):
        plot_values.append([])
        for j in range(3):
            plot_values[i].append([])

    for model_id, path in enumerate(path_list):
        headers, elements = draw_utils.read_csv(path, has_header=None)
        plot_x_list = [[] for _ in range(n_labels)]
        plot_y_list = [[] for _ in range(n_labels)]

        for lid, element in enumerate(elements):
            if element[0].startswith("unused") or "DP+" in element[0]:
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
            # print(f"[Note]{labels_list[i]}: plot_x_list:{plot_x_list[i]}\t plot_y_list:{plot_y_list[i]}")

        # sort x ticks
        y_max = 0
        for i in range(n_labels):
            x_val, y_val = plot_x_list[i], plot_y_list[i]
            idx = sorted(range(len(x_val)), key=x_val.__getitem__)
            plot_x_list[i] = [x_val[i] for i in idx]
            plot_y_list[i] = [y_val[i] for i in idx]
            y_max = max(np.max(y_val), y_max)

        print(f"[Note]plot_x_list:{plot_x_list}")
        print(f"[Note]plot_y_list:{plot_y_list}")
        esti_x_list = plot_x_list[-1]
        # select the min

        esti_y_list = []
        for i in range(len(esti_x_list)):
            esti_y_list.append(min([plot_y_list[0][i], plot_y_list[1][i], plot_y_list[2][i], plot_y_list[3][i]]))
        print("[Note]esti_x_list:", esti_x_list)
        print("[Note]esti_y_list:", esti_y_list)
        plot_x_list.append(esti_x_list)
        plot_y_list.append(esti_y_list)
        plot_values[0][model_id] = (plot_x_list, plot_y_list)

    xlabels = "CPU Memory (GB)"
    ylabels = "Epoch Time (s)"
    draw_utils.plot_main_exp(
        elements=plot_values,
        label_list=map_label_list + ["APT"],
        xlabels=xlabels,
        ylabels=ylabels,
        subfig_title=subfig_title,
        save_path="osdi_figs/main_exp_multi_machines.pdf",
    )

    # motivation fig
    """
    os.makedirs(output_prefix, exist_ok=True)
    draw_utils.plot_line(
        plot_x_list=plot_values[0][0][0],
        plot_y_list=plot_values[0][0][1],
        labels=map_label_list + ["APT"],
        # xticks=list(range(9)),
        # yticks=list(range(0, math.ceil(y_max) + 5, 5)),
        # xlabels="GPU Cache Memory (GB)" if fixed == "nl" else "Number of Local Node Features",
        xlabels="CPU Memory (GB)",
        ylabels="Epoch Time (s)",
        legends_font_size=18,
        save_path=os.path.join("./tmp_fig/moti_multi_machines.pdf"),
    )
    """


def draw_mainexp(path_list=["./logs/ap/Nov15_single_machine_stable.csv"], fixed="nl", output_prefix="./outputs/figures/single-machine/"):
    assert fixed in ["nl", "gpu_cache_mem"]
    varied = "gpu_cache_mem" if fixed == "nl" else "nl"
    data_dicts = {}
    fig_key_set = set()
    sys_set = set()

    from analysis_costmodel import costmodel_estimate

    esti_dict = costmodel_estimate()
    for path in path_list:
        headers, elements = draw_utils.read_csv(path, has_header=None)

        for lid, element in enumerate(elements):
            # decode tag
            tag_list = element[0].split("_")

            graph = tag_list[0]
            offset = 1 if "variance" in element[0] else 0
            method = tag_list[2 + offset]

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
    label_list = ori_label_list
    min_len = 100000

    xlabels = "GPU Cache Memory (GB)" if fixed == "nl" else "Number of Local Node Features"
    ylabels = "Epoch Time (s)"
    # init a 2d list
    graph_list = ["papers", "friendster", "igbfull"]
    graph_short_list = ["PS", "FS", "IH"]
    model_list = ["SAGE", "GCN", "GAT"]
    elements = []
    subfig_title = []
    for i in range(1):
        elements.append([])
        for j in range(3):
            elements[i].append([])

    for i in range(3):
        subfig_title.append([])
        for j in range(3):
            subfig_title[i].append(f"{graph_short_list[i]}-{model_list[j]}")

    for i in range(3):
        # draw one fig for 3 subgraph a row
        for j in range(3):
            fig_key = (graph_list[i], model_list[j], "metis", -1)

            print(f"[Note]fig_key:{fig_key}")
            esti_ret = esti_dict[fig_key[:2]]
            print("[Note]esti_ret:", esti_ret)
            plot_x_list = []
            plot_y_list = []
            for sys in label_list:
                key = (*fig_key, sys)
                if key not in data_dicts:
                    continue
                x_list = data_dicts[key][0]
                y_list = data_dicts[key][1]
                min_len = min(min_len, len(x_list))
                # filter_x_list, filter_y_list = filter_list(x_list, y_list)
                print(f"[Note]x_list:{x_list}\t y_list:{y_list}")
                # print(f"[Note]key:{key}\t data_dicts:{filter_x_list}\t filter_y_list:{filter_y_list}")
                plot_x_list.append(x_list)
                plot_y_list.append(y_list)

            # add esti
            esti_x_list = plot_x_list[-1][:min_len]
            esti_y_list = [plot_y_list[labels_to_idx(esti_ret[idx])][idx] for idx in esti_x_list]
            print("[Note]esti_x_list:", esti_x_list)
            print("[Note]esti_y_list:", esti_y_list)
            plot_x_list.append(esti_x_list)
            plot_y_list.append(esti_y_list)
            plot_x_list = [sub_list[:min_len] for sub_list in plot_x_list]
            plot_y_list = [sub_list[:min_len] for sub_list in plot_y_list]
            # plot_x_list.append(plot_x_list[-1])
            os.makedirs(output_prefix, exist_ok=True)
            # elements[i][j] = (plot_x_list, plot_y_list)
            elements[0][j] = (plot_x_list, plot_y_list)

            # single fig for a graph*model
            """
            draw_utils.plot_line(
                plot_x_list=plot_x_list,
                plot_y_list=plot_y_list,
                labels=label_list,
                # xticks=list(range(9)),
                # yticks=[0, 5, 10, 15, 20],
                xlabels="GPU Cache Memory (GB)" if fixed == "nl" else "Number of Local Node Features",
                ylabels="Epoch Time (s)",
                legends_font_size=18,
                save_path=os.path.join(output_prefix, f"moti_{fig_key[0]}_{fig_key[1]}_{fig_key[2]}_{fig_key[3]}.pdf"),
            )
            """
        # draw per row (3subfigs)
        draw_utils.plot_main_exp(
            elements=elements,
            label_list=map_label_list + ["APT"],
            xlabels=xlabels,
            ylabels=ylabels,
            subfig_title=[subfig_title[i]],
            save_path=f"osdi_figs/main_exp_row{i}.pdf",
        )


def draw_accuracy(
    path_prefix="./outputs/accuracy",
    filter_model="SAGE",
    filer_dataset="papers",
    filter_worldsize_list=[4, 16],
    time_list=[[16.95546, 19.19685], [20.5225, 67.43421]],
):
    # labels_list = [["GDP", "DNP", "SNP", "NFP", "DGL"], ["GDP", "DNP", "SNP", "NFP", "DistDGL"]]
    labels_list = [map_label_list + ["DGL"], map_label_list + ["DistDGL"]]
    # plot_x_list = []
    # plot_y_list = []
    # labels = []
    """
    to_plot_values = []
    for i in range(1):
        to_plot_values.append([])
        for j in range(len(filter_worldsize_list)):
            to_plot_values[i].append([])
    """
    for fig_id, filter_worldsize in enumerate(filter_worldsize_list):
        plot_x_list = [[] for _ in range(5)]
        plot_y_list = [[] for _ in range(5)]
        for file in sorted(os.listdir(path_prefix)):
            file_split = file.split("_")
            model, sys, dataset, num_workers = file_split
            num_workers = num_workers.split(".")[0]
            # print(f"[Note]model:{model}\t sys:{sys}\t dataset:{dataset}\t num_workers:{num_workers}")
            # filter by model, dataset, world_size
            if filter_model not in model or filer_dataset not in dataset or str(filter_worldsize) not in num_workers:
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
            sys_id = labels_to_idx(sys)
            print(f"[Note]sys_id:{sys_id}\t sys:{sys}")
            print(f"[Note]plot_x:{plot_x}\t plot_y:{plot_y}")
            plot_x_list[sys_id] = plot_x
            plot_y_list[sys_id] = plot_y
            # labels.append(sys)

        # print(f"[Note]plox_x_list:{np.array(plot_x_list).shape}\t plot_y_list:{np.array(plot_y_list).shape}")
        # plot_x_list = np.array(plot_x_list)
        # plot_y_list = np.array(plot_y_list)
        ymax = max([max(sub_list) for sub_list in plot_y_list]) + 0.005
        draw_utils.plot_line(
            plot_x_list=plot_x_list,
            plot_y_list=plot_y_list,
            labels=labels_list[fig_id],
            yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            xlabels="Number of epochs",
            ylabels="Accuracy",
            save_path=f"./tmp_fig/numepoch_{filter_model}_{filer_dataset}_{filter_worldsize}.pdf",
            legends_font_size=20,
            axhyline=ymax,
        )

        plot_x_time_list = []
        plot_y_time_list = []
        dp_list = plot_x_list[0]
        dgl_list = plot_x_list[-1]
        plot_x_time_list.append([v * time_list[fig_id][0] for v in dp_list])
        plot_x_time_list.append([v * time_list[fig_id][1] for v in dgl_list])

        plot_y_time_list.append(plot_y_list[0])
        plot_y_time_list.append(plot_y_list[-1])

        # xmax = [max(sub_list) for sub_list in plot_x_time_list]
        ymax = max([max(sub_list) for sub_list in plot_y_time_list]) + 0.01
        # legend column-spacing=1.2 legend box for world_size = 16
        xticks = [0, 1000, 2000] if filter_worldsize == 4 else [0, 2000, 4000]
        draw_utils.plot_line(
            plot_x_list=plot_x_time_list,
            plot_y_list=plot_y_time_list,
            labels=[labels_list[fig_id][0], labels_list[fig_id][-1]],
            xticks=xticks,
            yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            xlabels="Time (s)",
            ylabels="Accuracy",
            set_line_id=[0, 4],
            save_path=f"./tmp_fig/Time_{filter_model}_{filer_dataset}_{filter_worldsize}.pdf",
            legends_font_size=24,
            axhyline=ymax,
        )

        plot_x_list = []
        plot_y_list = []
        labels = []


def draw_micro(
    path_list=["./logs/ap/Nov15_single_machine_stable.csv"],
    fixed="nl",
    output_prefix="./outputs/figures/single-machine/",
    additional_key=None,
    select_key=None,
):
    assert fixed in ["nl", "gpu_cache_mem"]
    varied = "gpu_cache_mem" if fixed == "nl" else "nl"
    data_dicts = {}
    fig_key_set = set()
    sys_set = set()

    if additional_key == "hidden_dim":
        additional_id = 5
    elif additional_key == "input_dim":
        additional_id = 4
    elif additional_key == "fanout":
        additional_id = 7

    for path in path_list:
        headers, elements = draw_utils.read_csv(path, has_header=None)

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
                nl = int(tag_list[6 + offset][2:-4])
            else:
                nl = int(tag_list[6 + offset][2:-3])

            aux_key = element[additional_id]
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
        n_labels = len(label_list)
        # one dataset, model a graph
        plot_values = []
        for i in range(1):
            plot_values.append([])
            for j in range(2):
                plot_values[i].append([])

        subfig_title = [["Input dim 128", "Input dim 512"]]
        select_fig_key = []
        for sk in select_key:
            for fig_key in fig_key_set:
                if int(fig_key[-1]) == sk:
                    select_fig_key.append(fig_key)
        print(f"[Note]select_fig_key:{select_fig_key}")

        for fig_id, fig_key in enumerate(select_fig_key):
            print(f"[Note]fig_key:{fig_key}")
            # print(f"[Note]fig_key[-1]:{int(fig_key[-1])}\t select_key:{select_key}\t flag:{int(fig_key[-1]) not in select_key}")
            print(f"[Note]fig_key:{fig_key}\t fig_id:{fig_id}\t select_key:{select_key[fig_id]}")
            # subfig_title[0].append(fig_key[-1])
            plot_x_list = [[] for _ in range(n_labels)]
            plot_y_list = [[] for _ in range(n_labels)]
            for sys_id, sys in enumerate(label_list):
                key = (*fig_key, sys)
                if key not in data_dicts:
                    continue
                x_list = data_dicts[key][0]
                y_list = data_dicts[key][1]
                filter_x_list, filter_y_list = filter_list(x_list, y_list)
                print(f"[Note]key:{key}\t data_dicts:{filter_x_list}\t filter_y_list:{filter_y_list}")
                plot_x_list[sys_id] = filter_x_list
                plot_y_list[sys_id] = filter_y_list

            print(f"[Note]plot_x_list:{plot_x_list}")
            print(f"[Note]plot_y_list:{plot_y_list}")

            plot_values[0][fig_id] = (plot_x_list, plot_y_list)

        print(f"[Note]plot_values:{plot_values}")
        draw_utils.plot_main_exp(
            elements=plot_values,
            label_list=["GDP", "DNP", "SNP", "NFP"],
            xlabels="GPU Cache Memory (GB)" if fixed == "nl" else "Number of Local Node Features",
            ylabels="Epoch Time (s)",
            subfig_title=subfig_title,
            save_path=f"./osdi_figs/micro_exp_{additional_key}.pdf",
        )
        """
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
        """


if __name__ == "__main__":
    # draw_fig list
    # 1. sanity check
    # 2. main exp single machine
    # 3. main exp multi machine
    # 4. micro exp varying hidden dim, input dim, fanout
    # 5. cost model accuracy
    draw_sanity_check = True
    draw_main_exp_single_machine = False
    draw_main_exp_multi_machine = True
    draw_micro_exp = False
    draw_cost_model_accuracy = False

    fixed_list = ["nl", "gpu_cache_mem"]

    if draw_sanity_check:
        # sanity check
        draw_accuracy(filter_worldsize_list=[4, 16])

    if draw_main_exp_single_machine:
        single_machine_path_list = []
        single_machine_path_list.append("./outputs/speed/single_machine/papers_w4_metis.csv")
        single_machine_path_list.append("./outputs/speed/single_machine/friendster_w4_metis.csv")
        single_machine_path_list.append("./outputs/speed/single_machine/igbfull_w4_metis.csv")
        draw_mainexp(path_list=single_machine_path_list, fixed="nl", output_prefix="./tmp_fig")
    # multi-machines
    if draw_main_exp_multi_machine:
        multi_machine_path_prefix = "./outputs/speed/multi_machine/"
        multi_machine_files = os.listdir(multi_machine_path_prefix)
        path_list = []
        for file in multi_machine_files:
            if "_10_10_10" in file:
                path_list.append(os.path.join(multi_machine_path_prefix, file))
        print(f"[Note]path_list:{path_list}")
        # permute path_list
        keys = ["papers", "friendster", "igbfull"]
        permute_path_list = []
        for key in keys:
            for path in path_list:
                if key in path:
                    permute_path_list.append(path)
                    break
        print(f"[Note]permute_path_list:{permute_path_list}")
        draw_multimachines_mainexp(path_list=path_list, output_prefix="../outputs/figures/multi-machine-Dec-5")

    if draw_micro_exp:
        # micro on hidden dim
        # single_machine_path_list = []
        single_machine_path_list = ["./outputs/micro/varying_fanout.csv"]
        draw_micro(path_list=single_machine_path_list, fixed="nl", output_prefix="./outputs/figures/micro/vary_fanout", additional_key="fanout")

        # single_machine_path_list = ["./outputs/micro/varying_hidden_dim.csv"]
        draw_micro(
            path_list=single_machine_path_list,
            fixed="nl",
            output_prefix="./outputs/figures/micro/vary_fanout",
            additional_key="hidden_dim",
            select_key=[64, 256],
        )

        single_machine_path_list = ["./outputs/micro/varying_input_dim.csv"]
        draw_micro(
            path_list=single_machine_path_list,
            fixed="nl",
            output_prefix="./outputs/figures/micro/vary_fanout",
            additional_key="input_dim",
            select_key=[128, 512],
        )

    if draw_cost_model_accuracy:
        from analysis_costmodel import draw_costmodel_acc

        draw_costmodel_acc(keyword="friendster_w4_metis_SAGE", gpu_mem=4)
