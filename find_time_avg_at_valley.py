from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy

import utils


def find_time_avg_at_valley(raws: np.ndarray, tags: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """왼쪽, 오른쪽 센서의 시간 평균을 구하는 함수

    Args:
        raws (np.ndarray) (time, 198): raw 데이터
        tags (pd.DataFrame) (time, 5): tag 정보 (trial, id, action, fog, tray)

    Returns:
        np.ndarray (99,): valley에서 왼쪽 센서 값들 시간 축 평균
        np.ndarray (99,): valley에서 오른쪽 센서 값들 시간 축 평균
    """
    grps = split_grps(tags)

    lsensors = []
    rsensors = []
    for g in grps:
        raws_g = raws.iloc[g].values
        left = raws_g[:, :99]  # (time, 99)
        right = raws_g[:, 99:]  # (time, 99)
        left_mean = left.mean(axis=1)  # (time,)
        right_mean = right.mean(axis=1)  # (time,)

        left_valley_idxs = find_valleys(left_mean)[1:-1]  # [1, -1]: 양 끝 점이 잘려서 valley라고 잘못 인식하는 것 방지
        right_valley_idxs = find_valleys(right_mean)[1:-1]

        lsensors.extend(left[left_valley_idxs])
        rsensors.extend(right[right_valley_idxs])

    l_time_mean = np.sum(lsensors, axis=0) / len(lsensors)
    r_time_mean = np.sum(rsensors, axis=0) / len(rsensors)

    return l_time_mean, r_time_mean


def split_grps(tags: pd.DataFrame) -> List[np.ndarray]:
    """연속적인 걸음 그룹을 나누는 함수

    Args:
        tags (pd.DataFrame) (time, 5): tag 정보 (trial, id, action, fog, tray)

    Returns:
        List[np.ndarray]: 그룹 index의 list
    """

    tags_reset = tags.reset_index(drop=True)

    # action 0: 기립, 1: 직선, 88: 착석, 99: 지시불이행 / fog 0: 정상
    walk_idxs = tags_reset[(tags_reset["action"].isin(range(2, 9))) & (tags_reset["fog"] == 0)].index.values
    diff = np.diff(walk_idxs)
    grps = np.split(walk_idxs, np.where(diff != 1)[0] + 1)

    return grps


def find_valleys(arr: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        arr (np.ndarray) (time,)

    Returns:
        np.ndarray: arr에서 valley들 찾아서 index 반환
    """
    inv_m_normed = utils.normalize(-arr)
    peak_idxs = scipy.signal.find_peaks(inv_m_normed, height=0.8, distance=30)[0]
    return peak_idxs