import numpy as np

def get_transition_times(t, Vm, UD_threshold, DU_threshold):
    """
    2 thresholds strategy for state transition characterization
    """
    down_flag = False
    if Vm[0]>UD_threshold[0]:
        down_state = True
    UD_transitions, DU_transitions = [], [],
    for i in range(len(Vm)):
        if Vm[i]>DU_threshold[i] and down_flag:
            DU_transitions.append(t[i])
            down_flag = False
        if Vm[i]<UD_threshold[i] and not down_flag:
            UD_transitions.append(t[i])
            down_flag = True
    return np.array(UD_transitions), np.array(DU_transitions)

