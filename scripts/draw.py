import numpy as np
import itertools
import csv
import os
import draw_utils


def draw_motivation_fig(path_list=["./logs/ap/Nov15_single_machine_stable.csv"]):
    
    data_dicts = {}
    for path in path_list:
        headers, elements = draw_utils.read_csv(path, has_header=None)


        

        for lid, element in enumerate(elements):
            # decode tag
            tag_list = element[0].split("_")
            graph = tag_list[0]
            sys = tag_list[2]
            model = tag_list[3]
            gpu_cache_mem = int(tag_list[5][2:-2])

            epoch_time = float(element[-1]) / 1000.0
            # (gpu_cache_mem, epoch_time)
            key = (graph, model, sys)
            if key not in data_dicts:
                data_dicts[key] = [[], []]
            print(f"[Note] Insert key:{key} values:{gpu_cache_mem}, {epoch_time}")
            while len(data_dicts[key][0]) <= gpu_cache_mem:
                data_dicts[key][0].append(len(data_dicts[key][0]))
                data_dicts[key][1].append(-1)
                
            data_dicts[key][0][gpu_cache_mem] = gpu_cache_mem
            data_dicts[key][1][gpu_cache_mem] = epoch_time

    key_lists = list(data_dicts.keys())
    for key in key_lists:
        print(f"[Note]data dict key: {key}")
    dataset_list = ["papers", "friendster"]
    labels_list = ["DP", "NP", "SP", "MP"]
    models_list = ["SAGE", "GCN", "GAT"]
    # one dataset, model a graph
    for ds in dataset_list:
        for models in models_list:
            plot_x_list = []
            plot_y_list = []
            for label in labels_list:
                key = (ds, models, label)
                print(f"[Note]key:{key}\t data_dicts[key]:{data_dicts[key]}")
                plot_x_list.append(data_dicts[key][0])
                plot_y_list.append(data_dicts[key][1])

            draw_utils.plot_line(
                plot_x_list=plot_x_list,
                plot_y_list=plot_y_list,
                labels=labels_list,
                xticks=list(range(9)),
                xlabels="GPU Cache Memory (GB)",
                ylabels="Epoch Time (s)",
                legends_font_size=18,
                save_path=f"./scripts/figures/motivation/Nov15_motivation_{ds}_{models}.png",
            )


def draw_accuracy(path_prefix="./outputs/accuracy"):
    plot_x_list = []
    plot_y_list = []
    labels = []
    for file in sorted(os.listdir(path_prefix)):
        file_split = file.split("_")
        model = file_split[0]
        sys = file_split[1]
        num_workers = int(file_split[-1][0])
        if num_workers != 8:
            continue
        path = os.path.join(path_prefix, file)
        print(f"[Note] model:{model}\t sys:{sys}\t num_workers:{num_workers}")
        # read txt
        plot_x = []
        plot_y = []
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                line_split = line.split(" ")
                plot_y.append(float(line_split[-1]))

        plot_x = list(range(len(plot_y)))

        plot_x_list.append(plot_x)
        plot_y_list.append(plot_y)
        labels.append(sys)
        if len(plot_x_list) == 4:
            draw_utils.plot_line(
                plot_x_list=plot_x_list,
                plot_y_list=plot_y_list,
                labels=labels,
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                xlabels="Numbe of epochs",
                ylabels="Accuracy",
                save_path=f"./outputs/figures/accuracy_{model}.png",
            )
            plot_x_list = []
            plot_y_list = []
            labels = []


def draw_ap_result(path="./logs/ap/9_3_ver1.csv"):
    headers, elements = draw_utils.read_csv(path)
    # [papers, friendster]
    # [TODO]add dataset igb-large
    num_graphs = 1
    # [DP, NP, SP, MP, DP2, SP2]
    num_systems = 6
    # [Sampling, Loading, Training]
    num_stages = 3
    # [0,1,2,3,4,5] GB
    num_cache = 6
    num_rows = len(elements)
    num_cols = [len(e) for e in elements]
    all_elements = [[[] for _ in range(num_stages)] for j in range(num_systems)]
    for sys_id in range(num_systems):
        for stage_id in range(num_stages):
            for cac_id in range(num_cache):
                all_elements[sys_id][stage_id].append(0)

    def system_to_num(system):
        if system == "DP":
            return 0
        elif system == "NP":
            return 1
        elif system == "SP":
            return 2
        elif system == "MP":
            return 3
        elif system == "DP2":
            return 4
        elif system == "NP2":
            return 5
        else:
            raise ValueError(f"[Error] system:{system}")

    for rid, row in enumerate(elements):
        tag = row[0].split("_")
        graph = tag[0]
        system = tag[-1]
        cache_mem = int(float(row[5][:-2]))
        sampling_time, loading_time, training_time = [float(row[i]) for i in range(-4, -1)]
        print(
            f"[Note] graph:{graph}\t system:{system}\t cache_mem:{cache_mem}\t sampling_time:{sampling_time}\t loading_time:{loading_time}\t training_time:{training_time}"
        )

        all_elements[system_to_num(system)][0][cache_mem] = sampling_time
        all_elements[system_to_num(system)][1][cache_mem] = loading_time
        all_elements[system_to_num(system)][2][cache_mem] = training_time
        print(f"[Note] rid:{rid}\t to_print_row = {num_systems * num_cache}")
        if (rid + 1) % (num_systems * num_cache) == 0:
            draw_utils.draw_figure_group_stacked_bar(
                all_elements,
                save_path=f"./temp_{graph}.png",
                xlabel="Cache memory (GB)",
                ylabel="Time (s)",
                xtickslabel=[f"{i}GB" for i in range(num_cache)],
                # yticks=[0, 5, 10, 15, 20],
                labels=["DP", "NP", "SP", "MP", "DP2", "NP2"],
            )


def draw_multi_machines_varied_cpu_memory(path="./logs/ap/multi_machines_cpu_mem.csv"):
    headers, elements = draw_utils.read_csv(path, has_header=None)
    print(f"[Note]headers:{headers}\t elements:{len(elements)} * {len(elements[0])}")
    val_dict = {}
    max_cache_mem = 0
    for element in elements:
        cache_mem = int(element[5][0])
        max_cache_mem = max(max_cache_mem, cache_mem)
        e0_key = element[0].split("_")
        method = e0_key[-1]
        cpu_mem = int(e0_key[-2])
        key = (method, cpu_mem, cache_mem)
        total_time = float(element[-1])
        print(f"[Note]key:{key}\t total_time:{total_time}")
        if key not in val_dict:
            val_dict[key] = []
        val_dict[key].append(total_time)

    def to_line_id(key):
        line_id = key[1] - 4 + 4 * (key[0] == "NP")
        return line_id

    total_line = 8
    plot_x_list = []
    plot_y_list = []
    labels = []
    for line_id in range(total_line):
        plot_x_list.append(list(range(max_cache_mem + 1)))
        plot_y_list.append([-1 for _ in range(max_cache_mem + 1)])
        labels.append("TBD")

    for key, value in val_dict.items():
        max_value = max(value)
        min_value = min(value)
        mea_value = sum(value) / len(value)
        err = (max_value - min_value) / min_value
        line_id = to_line_id(key)
        sys = "DP" if line_id < 4 else "NP"
        cpu_cache_mem = key[1]
        labels[line_id] = f"{sys} {cpu_cache_mem}/8"
        print(f"[Note]key:{key}\t line_id:{line_id}\t max_value:{max_value}\t min_value:{min_value}\t mea_value:{mea_value}\t err:{err}")
        plot_y_list[line_id][key[2]] = mea_value / 1000

    for line_id in range(total_line):
        print(f"[Note]line: {labels[line_id]}\t value_y:{plot_y_list[line_id]}")

    draw_utils.plot_line(
        plot_x_list=plot_x_list,
        plot_y_list=plot_y_list,
        labels=labels,
        xlabels="GPU Cache Mem.",
        ylabels="Epoch Time (s)",
        legends_font_size=18,
        save_path="./scripts/figures/Oct09_multi_machines_varied_cpumem.png",
        fig_size=(8, 4.8),
    )


if __name__ == "__main__":
    # draw_figure()
    # draw_figure(input_path="./data/data2.csv", tag="fig2", header_offset=1)
    # draw_cost_model(input_path="./logs/cost_model_greedy4.csv", save_dir="./scripts/figures")
    # draw_ap_result(path="./logs/ap/multi_machines.csv")
    # draw_multi_machines_varied_cpu_memory()

    draw_motivation_fig(path_list=["./logs/ap/Nov15_single_machine_stable.csv", "./logs/ap/Nov17-single-machine-SP.csv"])
    # draw_accuracy()
