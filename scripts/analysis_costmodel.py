import os
import statistics
import numpy as np
import csv


def tag_match(ori_tag, to_match_tag):
    tag_list = ori_tag.split("_")
    match_flag = True
    for tag in tag_list:
        if "variance" in tag or "nl" in tag:
            continue
        if "t4" == tag:
            continue
        if tag not in to_match_tag:
            # print(f"[Note]{tag} not in {to_match_tag}")
            match_flag = False
            break
    if match_flag:
        print(f"[Note]ori_tag:{ori_tag}\t to_match_tag:{to_match_tag}")
    return match_flag


def add_newlines_to_csv(old_file_path, new_file_path):
    from draw_utils import read_csv

    _, old_lines = read_csv(old_file_path, has_header=False)
    _, new_lines = read_csv(new_file_path, has_header=False)
    num_old_lines = len(old_lines)
    for new_line in new_lines:
        tag = new_line[0]
        # print(f"[Note]tag:{tag}\t new_line:{new_line}")
        for i in range(num_old_lines):
            if tag_match(tag, old_lines[i][0]):
                old_lines[i] = new_line
                break
    # save csv to file
    with open("tmp.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(old_lines)


def str_to_list(str, separator=","):
    str_list = str.strip().split(separator)[:-1]
    str_list = [float(val) for val in str_list]
    return str_list


def decode_val_list(val_list):
    sample_time = val_list[0]
    reshuffle_time = val_list[-1]
    featloading_time = val_list[1:-1]
    return sample_time, reshuffle_time, featloading_time


def get_realtime_records(keywords, path="./outputs/speed/single_machine_dec02/"):
    filename = os.listdir(path)
    rec_dict = {}
    for file in filename:
        if keywords not in file or "metis" not in file or "w4" not in file:
            continue
        file_path = os.path.join(path, file)
        print("[Note]file_path:", file_path)
        from draw_utils import read_csv

        _, lines = read_csv(file_path, has_header=False)
        for line in lines:
            tag = line[0]
            offset = 1 if "variance" in tag else 0
            # useful key
            tag_list = tag.split("_")
            sys = tag_list[4 + offset]
            model = tag_list[5 + offset]
            cm = int(tag_list[-1][2:-2])
            key = (sys, model)

            # useful values
            # sampling_time = float(line[1+
            sampling_time = float(line[-4])
            loading_time = float(line[-3])
            training_time = float(line[-2])
            epoch_time = float(line[-1])

            if key not in rec_dict:
                rec_dict[key] = [[], [], []]
            rec_dict[key][0].append(sampling_time)
            rec_dict[key][1].append(training_time)
            rec_dict[key][2].append(loading_time)

            # rec_dict[key][2].append([cm, loading_time])
    return rec_dict


def return_list_max_min(list):
    return f"max:{max(list):.2f}\t min:{min(list):.2f}\t R:{max(list)/min(list):.2f}"


def extract_from_list(rec_list):
    sampling_time_list, training_time_list, feat_loading_list = rec_list
    # print max min variance
    print(f"[INFO]sampling_time_list:{return_list_max_min(sampling_time_list)}")
    print(f"[INFO]training_time_list:{return_list_max_min(training_time_list)}")
    sampling_time = statistics.mean(sampling_time_list)
    training_time = statistics.mean(training_time_list)
    return sampling_time, training_time, feat_loading_list


def costmodel_estimate(path_prefix="./costmodel/epoch10"):
    esti_dict = {}
    filename = os.listdir(path_prefix)
    for file in filename:
        file_path = os.path.join(path_prefix, file)
        keyword = file.split(".")[0]
        keyword_list = keyword.split("_")
        model = keyword_list[-1]
        dataset = keyword_list[1]
        print(f"[Note]file:{file}, model:{model}\t dataset:{dataset}")
        num_batches_per_epoch = -1
        if dataset == "friendster":
            num_batches_per_epoch = 160
        elif dataset == "igbfull":
            num_batches_per_epoch = 400
        elif dataset == "papers":
            num_batches_per_epoch = 294
        else:
            raise NotImplementedError
        min_feat_load_time = num_batches_per_epoch * 2.4
        with open(file_path, "r") as f:
            lines = f.readlines()
            print(f"[Note]#lines:{len(lines)}")
            for line in lines:
                val_list = str_to_list(line)
                num_vals = len(val_list)
                vals_per_ps = int(num_vals // 4)
                num_cache_mem = vals_per_ps - 2
                # print(f"[Note]#vals:{num_vals}\t vals_per_ps:{vals_per_ps}")
                dp_vals = val_list[0:vals_per_ps]
                np_vals = val_list[vals_per_ps : 2 * vals_per_ps]
                sp_vals = val_list[2 * vals_per_ps : 3 * vals_per_ps]
                mp_vals = val_list[3 * vals_per_ps : 4 * vals_per_ps]
                esti_ret = []
                for i in range(num_cache_mem):
                    dp_vals[i + 1] = max(dp_vals[i + 1], min_feat_load_time)
                    np_vals[i + 1] = max(np_vals[i + 1], min_feat_load_time)
                    sp_vals[i + 1] = max(sp_vals[i + 1], min_feat_load_time)
                    mp_vals[i + 1] = max(mp_vals[i + 1], min_feat_load_time)
                    dp_epoch_time = dp_vals[0] + dp_vals[-1] + dp_vals[i + 1]
                    np_epoch_time = np_vals[0] + np_vals[-1] + np_vals[i + 1]
                    sp_epoch_time = sp_vals[0] + sp_vals[-1] + sp_vals[i + 1]
                    mp_epoch_time = mp_vals[0] + mp_vals[-1] + mp_vals[i + 1]
                    # select the min epoch time return (dp, np, sp, mp)
                    epoch_time_list = [(dp_epoch_time, "DP"), (np_epoch_time, "NP"), (sp_epoch_time, "SP"), (mp_epoch_time, "MP")]
                    epoch_time_list.sort(key=lambda x: x[0])
                    esti_ret.append(epoch_time_list[0][1])
                    print(f"[Note]dp:{dp_vals[0]}\t dp_feat:{dp_vals[i+1]}\t dp_reshuffle:{dp_vals[-1]}")
                    print(f"[Note]np:{np_vals[0]}\t np_feat:{np_vals[i+1]}\t np_reshuffle:{np_vals[-1]}")
                    # print the min key and epoch time
                    print(
                        f"[Note]cache_mem:{i}\t {epoch_time_list[0][1]}:{epoch_time_list[0][0]:.2f}\t {epoch_time_list[1][1]}:{epoch_time_list[1][0]:.2f}\t {epoch_time_list[2][1]}:{epoch_time_list[2][0]:.2f}\t {epoch_time_list[3][1]}:{epoch_time_list[3][0]:.2f}\t"
                    )
                esti_dict[(dataset, model)] = esti_ret
    return esti_dict


def read_costmodel(path_prefix="./costmodel/epoch10", keywords="friendster"):
    realtime_rec_dict = get_realtime_records(keywords=keywords)

    # print(f"[Note]realtime_rec_dict:{realtime_rec_dict}")
    filename = os.listdir(path_prefix)
    for file in filename:
        if keywords not in file:
            continue
        file_path = os.path.join(path_prefix, file)
        model = file.split("_")[-1].split(".")[0]
        keyword = file.split(".")[0]
        print(f"[Note]file:{file}, model:{model}")
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                val_list = str_to_list(line)
                num_vals = len(val_list)
                vals_per_ps = int(num_vals // 4)
                print(f"[Note]#vals:{num_vals}\t vals_per_ps:{vals_per_ps}")
                dp_vals = val_list[0:vals_per_ps]
                np_vals = val_list[vals_per_ps : 2 * vals_per_ps]
                sp_vals = val_list[2 * vals_per_ps : 3 * vals_per_ps]
                mp_vals = val_list[3 * vals_per_ps : 4 * vals_per_ps]

                real_dp_sampling_time, real_dp_training_time, real_dp_featloading_time = extract_from_list(realtime_rec_dict[("DP", model)])
                real_np_sampling_time, real_np_training_time, real_np_featloading_time = extract_from_list(realtime_rec_dict[("NP", model)])
                real_sp_sampling_time, real_sp_training_time, real_sp_featloading_time = extract_from_list(realtime_rec_dict[("SP", model)])
                real_mp_sampling_time, real_mp_training_time, real_mp_featloading_time = extract_from_list(realtime_rec_dict[("MP", model)])

                dp_sample_time, dp_reshuffle_time, dp_featloading_time = decode_val_list(dp_vals)
                np_sample_time, np_reshuffle_time, np_featloading_time = decode_val_list(np_vals)
                sp_sample_time, sp_reshuffle_time, sp_featloading_time = decode_val_list(sp_vals)
                mp_sample_time, mp_reshuffle_time, mp_featloading_time = decode_val_list(mp_vals)

                esti_dp_training_time = real_dp_training_time + dp_reshuffle_time
                esti_np_training_time = real_dp_training_time + np_reshuffle_time

                esti_sp_training_time = real_dp_training_time + sp_reshuffle_time
                esti_mp_training_time = real_dp_training_time + mp_reshuffle_time

                print(
                    f"[Note]esti_dp_sample_time:{dp_sample_time:.2f}\t esti_dp_training_time:{esti_dp_training_time:.2f}\t"
                )  # esti_dp_featloading_time:{esti_dp_featloading_time:.2f}")
                print(
                    f"[Note]real_dp_sampling_time:{real_dp_sampling_time:.2f}\t real_dp_training_time:{real_dp_training_time:.2f}\t"
                )  # real_dp_featloading_time:{real_dp_featloading_time}

                print(
                    f"[Note]esti_np_sample_time:{np_sample_time:.2f}\t esti_np_trianing_time:{esti_np_training_time:.2f}\t"
                )  # esti_np_featloading_time:{esti_np_featloading_time:.2f}")
                print(
                    f"[Note]real_np_sampling_time:{real_np_sampling_time:.2f}\t real_np_training_time:{real_np_training_time:.2f}\t"
                )  # real_np_featloading_time:{real_np_featloading_time}")

                print(
                    f"[Note]esti_sp_sample_time:{sp_sample_time:.2f}\t esti_sp_training_time:{esti_sp_training_time:.2f}\t"
                )  # esti_sp_featloading_time:{esti_sp_featloading_time:.2f}")
                print(
                    f"[Note]real_sp_sampling_time:{real_sp_sampling_time:.2f}\t real_sp_training_time:{real_sp_training_time:.2f}\t"
                )  # real_sp_featloading_time:{real_sp_featloading_time}")

                print(
                    f"[Note]esti_mp_sample_time:{mp_sample_time:.2f}\t esti_mp_training_time:{esti_mp_training_time:.2f}\t"
                )  # esti_mp_featloading_time:{esti_mp_featloading_time:.2f}")
                print(
                    f"[Note]real_mp_sampling_time:{real_mp_sampling_time:.2f}\t real_mp_training_time:{real_mp_training_time:.2f}\t"
                )  # real_mp_featloading_time:{real_mp_featloading_time}")

                # check feature loading
                print(f"[Note]esti_dp_featloading_time:{dp_featloading_time}")
                print(f"[Note]real_dp_featloading_time:{real_dp_featloading_time}")
                print(f"[Note]esti_np_featloading_time:{np_featloading_time}")
                print(f"[Note]real_np_featloading_time:{real_np_featloading_time}")
                print(f"[Note]esti_sp_featloading_time:{sp_featloading_time}")
                print(f"[Note]real_sp_featloading_time:{real_sp_featloading_time}")
                print(f"[Note]esti_mp_featloading_time:{mp_featloading_time}")
                print(f"[Note]real_mp_featloading_time:{real_mp_featloading_time}")

                # write csv to file
                output_path_prefix = "./outputs/costmodel/"
                output_file_csv = os.path.join(output_path_prefix, f"{keyword}.csv")
                os.makedirs(output_path_prefix, exist_ok=True)
                model_lists = ["DP", "NP", "SP", "MP"]
                num_models = len(model_lists)
                cache_mem = list(range(len(real_mp_featloading_time)))
                num_cache_mem = len(cache_mem)
                print(f"[Note]output_file_csv:{output_file_csv}")
                print(f"[Note]cache_mem:{cache_mem}")
                headers = [
                    "model",
                    "cache mem",
                    "Esti Sampling Time",
                    "Real Sampling Time",
                    "Err",
                    "Esti Featloading Time",
                    "Real Featloading Time",
                    "Err",
                    "Esti Training Time",
                    "Real Training Time",
                    "Err",
                ]

                esti_sampling_time_list = (
                    [dp_sample_time] * num_cache_mem
                    + [np_sample_time] * num_cache_mem
                    + [sp_sample_time] * num_cache_mem
                    + [mp_sample_time] * num_cache_mem
                )
                real_sampling_time_list = (
                    [real_dp_sampling_time] * num_cache_mem
                    + [real_np_sampling_time] * num_cache_mem
                    + [real_sp_sampling_time] * num_cache_mem
                    + [real_mp_sampling_time] * num_cache_mem
                )
                err_sampling_time_list = [
                    abs(esti_sampling_time_list[i] - real_sampling_time_list[i]) / real_sampling_time_list[i]
                    for i in range(len(esti_sampling_time_list))
                ]

                esti_featloading_time_list = (
                    dp_featloading_time[:num_cache_mem]
                    + np_featloading_time[:num_cache_mem]
                    + sp_featloading_time[:num_cache_mem]
                    + mp_featloading_time[:num_cache_mem]
                )
                real_featloading_time_list = (
                    real_dp_featloading_time[:num_cache_mem]
                    + real_np_featloading_time
                    + real_sp_featloading_time[:num_cache_mem]
                    + real_mp_featloading_time[:num_cache_mem]
                )

                err_featloading_time_list = [
                    abs(esti_featloading_time_list[i] - real_featloading_time_list[i]) / real_featloading_time_list[i]
                    for i in range(len(esti_featloading_time_list))
                ]

                esti_training_time_list = (
                    [esti_dp_training_time] * num_cache_mem
                    + [esti_np_training_time] * num_cache_mem
                    + [esti_sp_training_time] * num_cache_mem
                    + [esti_mp_training_time] * num_cache_mem
                )
                real_training_time_list = (
                    [real_dp_training_time] * num_cache_mem
                    + [real_np_training_time] * num_cache_mem
                    + [real_sp_training_time] * num_cache_mem
                    + [real_mp_training_time] * num_cache_mem
                )
                err_training_time_list = [
                    abs(esti_training_time_list[i] - real_training_time_list[i]) / real_training_time_list[i]
                    for i in range(len(esti_training_time_list))
                ]

                # print each list len
                print(f"[Note]esti_sampling_time_list:{len(esti_sampling_time_list)}")
                print(f"[Note]real_sampling_time_list:{len(real_sampling_time_list)}")
                print(f"[Note]err_sampling_time_list:{len(err_sampling_time_list)}")
                print(f"[Note]esti_featloading_time_list:{len(esti_featloading_time_list)}")
                print(f"[Note]real_featloading_time_list:{len(real_featloading_time_list)}")
                print(f"[Note]esti_training_time_list:{len(esti_training_time_list)}")
                print(f"[Note]real_training_time_list:{len(real_training_time_list)}")
                continue
                data_gen = zip(
                    esti_sampling_time_list,
                    real_sampling_time_list,
                    err_sampling_time_list,
                    esti_featloading_time_list,
                    real_featloading_time_list,
                    err_featloading_time_list,
                    esti_training_time_list,
                    real_training_time_list,
                    err_training_time_list,
                )
                with open(output_file_csv, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    for i, (
                        esti_sampling_time,
                        real_sampling_time,
                        err_sampling_time,
                        esti_featloading_time,
                        real_featloading_time,
                        err_featloading_time,
                        esti_training_time,
                        real_training_time,
                        err_training_time,
                    ) in enumerate(data_gen):
                        # write :2f
                        writer.writerow(
                            [
                                model_lists[int(i / num_cache_mem)],
                                cache_mem[int(i % num_cache_mem)],
                                f"{esti_sampling_time:.2f}",
                                f"{real_sampling_time:.2f}",
                                f"{err_sampling_time:.2f}",
                                f"{esti_featloading_time:.2f}",
                                f"{real_featloading_time:.2f}",
                                f"{err_featloading_time:.2f}",
                                f"{esti_training_time:.2f}",
                                f"{real_training_time:.2f}",
                                f"{err_training_time:.2f}",
                            ]
                        )


def draw_costmodel_acc(input_path_prefix="./outputs/costmodel", keyword="friendster_w4_metis", gpu_mem=4):
    file_list = os.listdir(input_path_prefix)
    for file in file_list:
        if keyword not in file:
            continue
        file_path = os.path.join(input_path_prefix, file)
        print(f"[Note]file_path:{file_path}")
        from draw_utils import read_csv

        headers, lines = read_csv(file_path, has_header=True)
        print(f"[Note]headers:{headers}")
        xticks = [0, 1, 3, 4, 6, 7, 9, 10]
        yvals = np.zeros((3, 8))
        for line in lines:
            sys = line[0]
            cache_mem = int(line[1])
            if gpu_mem != cache_mem:
                continue

            esti_sampling_time = float(line[2])
            real_sampling_time = float(line[3])
            # err_sampling_time = float(line[4])
            esti_featloading_time = float(line[5])
            real_featloading_time = float(line[6])
            # err_featloading_time = float(line[7])
            esti_training_time = float(line[8])
            real_training_time = float(line[9])
            # err_training_time = float(line[10])

            sys_id = sys_to_id(sys)
            real_sys_id = 2 * sys_id
            esti_sys_id = 2 * sys_id + 1
            yvals[0][esti_sys_id] = esti_sampling_time
            yvals[0][real_sys_id] = real_sampling_time
            yvals[1][esti_sys_id] = esti_featloading_time
            yvals[1][real_sys_id] = real_featloading_time
            yvals[2][esti_sys_id] = esti_training_time
            yvals[2][real_sys_id] = real_training_time

        # print("[Note]yvals:", yvals)
        from draw_utils import plot_stacked_bar

        yvals /= 1000
        labels = ["Sampling", "FeatLoading", "Training"]
        ylabels = "Epoch time (s)"
        xlabels = ["GDP", "DNP", "SNP", "NFP"]
        xlabelticks = [0.5, 3.5, 6.5, 9.5]
        plot_stacked_bar(
            elements=yvals,
            xticks=xticks,
            labels=labels,
            ylabels=ylabels,
            xlabelticks=xlabelticks,
            xlabels=xlabels,
            save_path="./osdi_figs/micro_costmodel.pdf",
        )


def sys_to_id(sys):
    if sys == "DP":
        return 0
    elif sys == "NP":
        return 1
    elif sys == "SP":
        return 2
    elif sys == "MP":
        return 3
    else:
        raise NotImplementedError


if __name__ == "__main__":
    # old_file_path = "./outputs/speed/single_machine/igbfull_w4_metis.csv"
    # new_file_path = "./outputs/speed/single_machine/NP_friendster_igb.csv"
    # add_newlines_to_csv(old_file_path=old_file_path, new_file_path=new_file_path)
    # check_sampling_all()
    # for keyword in ["papers", "friendster", "igbfull"]:
    # read_costmodel(path_prefix="./costmodel/epoch3", keywords="friendster")
    # costmodel_estimate()
    # read_costmodel(keywords="igbfull")
    # read_costmodel(keywords="papers")
    draw_costmodel_acc(keyword="friendster_w4_metis_SAGE", gpu_mem=4)
