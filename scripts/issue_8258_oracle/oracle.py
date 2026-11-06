"""Select diagnostic winners using tuning data, then report held-out timings."""
import statistics


def median_rounds(rounds):
    return statistics.median(statistics.median(values) for values in rounds)


def select_oracle(indices, timings, mapping):
    candidates = [(str(index), policy) for index in indices
                  for policy in ('baseline', 'intervention')]
    if not candidates:
        raise ValueError('No valid oracle candidates')
    index, policy = min(candidates, key=lambda item: (
        median_rounds(timings[mapping[f'{item[0]}:{item[1]}']]),
        int(item[0]), item[1] != 'baseline'))
    return f'{index}:{policy}'


def confirmed_ms(observation, selected):
    return median_rounds(observation['confirmation_ms'][observation['mapping'][selected]])


def require_matching_availability(statuses):
    available = {policy: status == 'compiled' for policy, status in statuses.items()}
    if len(set(available.values())) > 1:
        raise RuntimeError(f'Asymmetric compile availability: {statuses}')
    return all(available.values())
