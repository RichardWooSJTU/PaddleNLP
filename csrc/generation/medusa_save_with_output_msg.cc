#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include "paddle/extension.h"

#define MAX_TOKENS 512
#define MAX_BSZ 128

struct msgdata {
    long mtype;
    int mtext[MAX_TOKENS + MAX_BSZ + 1];   // stop_flag, bsz, tokens
};

void MedusaSaveOutMmsg(const paddle::Tensor& x,
                 const paddle::Tensor& accept_lengths,
                 const paddle::Tensor& insert_index_decoder,
                 const paddle::Tensor& not_need_stop,
                 int64_t rank_id) {
    if (rank_id > 0) return;
    auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
    int64_t *x_data = x_cpu.data<int64_t>();
    auto lengths_cpu = accept_lengths.copy_to(paddle::CPUPlace(), false);
    int* lengths_data = lengths_cpu.data<int>();
    auto insert_index_decoder_cpu = insert_index_decoder.copy_to(paddle::CPUPlace(), false);
    int* insert_index_decoder_data = insert_index_decoder_cpu.data<int>();


    static struct msgdata msg_sed;
    static key_t key = ftok("./", 1);
    static int msgid = msgget(key, IPC_CREAT | 0666);

    msg_sed.mtype = 1;
    bool not_need_stop_data = not_need_stop.data<bool>()[0];
    msg_sed.mtext[0] = not_need_stop_data ? 1 : -1;
    int num_decoders = x.shape()[0];
    int num_medusa = x.shape()[1];
    int i = 1;
    // for (int i = 2; i < bsz + 2; i++) {
    //     msg_sed.mtext[i] = (int)x_data[i - 2];
    // }
    for (int decoder_i = 0; decoder_i < num_decoders; decoder_i++) {
        msg_sed.mtext[i++] = -insert_index_decoder_data[decoder_i]; // 将bsz_i设置为负数以区分token_id
        for (int token_id = 0; token_id < lengths_data[decoder_i]; token_id++) {
            msg_sed.mtext[i++] = (int)x_data[decoder_i * num_medusa + token_id];
        }
    }
    if ((msgsnd(msgid, &msg_sed, (MAX_TOKENS + MAX_BSZ + 1) * 4, IPC_NOWAIT)) == -1) {
      printf("full msg buffer\n");
    }
    return;
}

PD_BUILD_OP(medusa_save_output)
    .Inputs({"x", "accept_lengths", "insert_index_decoder", "not_need_stop"})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(MedusaSaveOutMmsg));