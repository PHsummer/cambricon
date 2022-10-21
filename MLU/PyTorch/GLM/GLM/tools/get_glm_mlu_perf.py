import numpy


def get_glm_perf(file_name):
    total_time = 0.0
    train_counts = 0
    target_tokens = "elapsed time per iteration (ms): "
    with open(file_name , "rb") as fr:
        for line in fr.readlines():
              line = str(line)
              target_tokens_idx = line.find(target_tokens)
              if target_tokens_idx != -1:
                  train_counts += 1
                  time_idx = target_tokens_idx+len(target_tokens)
                  blank_idx = line[time_idx:].find(" ")
                  time = float(line[time_idx:time_idx+blank_idx])
                  total_time += time
        print(f"==== train_counts: {train_counts}, avg_time: {round((total_time/train_counts), 2)}ms")




if __name__ == '__main__':
    file_name = "./doc/glm-xxlarge-mlu-dp4-mp8.txt"
    get_glm_perf(file_name)
