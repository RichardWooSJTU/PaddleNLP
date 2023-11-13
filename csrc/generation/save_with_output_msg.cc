#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include "paddle/extension.h"

#define MAX_BSZ 256

struct msgdata {
    long mtype;
    int mtext[MAX_BSZ + 2];   // stop_flag, bsz, tokens
};

void SaveOutMmsg(const paddle::Tensor& x,
                 const paddle::Tensor& not_need_stop,
                 int64_t rank_id) {
    if (rank_id > 0) return;
    auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
    int64_t *x_data = x_cpu.data<int64_t>();
    static struct msgdata msg_sed;
    static key_t key = ftok("./", 1);
    static int msgid = msgget(key, IPC_CREAT | 0666);

    msg_sed.mtype = 1;
    bool not_need_stop_data = not_need_stop.data<bool>()[0];
    msg_sed.mtext[0] = not_need_stop_data ? 1 : -1;
    int bsz = x.shape()[0];
    msg_sed.mtext[1] = bsz;
    for (int i = 2; i < bsz + 2; i++) {
        msg_sed.mtext[i] = (int)x_data[i - 2];
    }
    if ((msgsnd(msgid, &msg_sed, (MAX_BSZ + 2) * 4, 0)) == -1) {
    //   printf("full msg buffer\n");
    }
    return;
}

PD_BUILD_OP(save_output)
    .Inputs({"x", "not_need_stop"})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutMmsg));