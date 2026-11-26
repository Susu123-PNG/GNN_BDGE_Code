"""Occupation skill profiles and market-instance construction."""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def _read_member(source, name):
    path = Path(source)
    if path.is_dir():
        matches = list(path.rglob(name))
        if not matches:
            raise FileNotFoundError(f"Missing {name} in {source}")
        return matches[0].open('r', newline='', encoding='utf-8-sig')
    with zipfile.ZipFile(path) as archive:
        candidates = [x for x in archive.namelist() if x.lower().endswith('/' + name.lower()) or x.lower() == name.lower()]
        if not candidates:
            raise FileNotFoundError(f"Missing {name} in {source}")
        contents = archive.read(candidates[0]).decode('utf-8-sig')
    return io.StringIO(contents)


def load_onet_profiles(source, k=30):
    # Merge the two core O*NET skills domains using the importance scale.
    ratings = {}
    frequency = {}
    for filename in ('essential_skills.csv', 'transferable_skills.csv'):
        with _read_member(source, filename) as handle:
            for row in csv.DictReader(handle):
                if row.get('Scale ID') != 'IM' or row.get('Not Relevant') == 'Y':
                    continue
                occupation = row.get('O*NET-SOC Code', '').strip()
                element = row.get('Element ID', '').strip()
                try:
                    value = float(row['Data Value'])
                except (KeyError, ValueError, TypeError):
                    continue
                if occupation and element:
                    ratings.setdefault(occupation, {})[element] = value
                    frequency[element] = frequency.get(element, 0) + 1
    elements = sorted(frequency, key=lambda s: (-frequency[s], s))[:k]
    if len(elements) < k:
        raise ValueError(f"Only {len(elements)} skills are available, expected {k}.")
    codes, vectors = [], []
    for occupation, items in sorted(ratings.items()):
        vector = np.array([items.get(key, np.nan) for key in elements], dtype=np.float32)
        if np.count_nonzero(np.isfinite(vector)) < k // 2:
            continue
        vector = np.where(np.isfinite(vector), vector, np.nanmean(vector))
        # The importance scale runs from 1 to 5.
        vector = np.clip((vector - 1.0) / 4.0, 0.0, 1.0)
        codes.append(occupation)
        vectors.append(vector)
    if len(vectors) < 10:
        raise ValueError('Insufficient occupation profiles after filtering.')
    return np.stack(vectors), {'source_mode': 'onet31', 'occupation_count': len(codes), 'skill_ids': elements,
                               'source_file': str(source), 'occupation_codes': codes}


def procedural_profiles(seed, k=30, occupation_count=200):
    # This input mode provides a self-contained test when a source archive is unavailable.
    rng = np.random.default_rng(seed)
    family = rng.uniform(0.08, 0.95, size=(12, k))
    membership = rng.integers(0, len(family), size=occupation_count)
    profiles = np.clip(family[membership] + rng.normal(0, 0.09, size=(occupation_count, k)), 0, 1)
    return profiles.astype(np.float32), {'source_mode': 'procedural', 'occupation_count': occupation_count,
                                         'skill_ids': [f'skill_{i+1:02d}' for i in range(k)], 'seed': seed}


def check_esco(esco_dir):
    # ESCO files are kept separate from the O*NET feature vector until a verified skill crosswalk exists.
    if not esco_dir:
        return {'esco_checked': False}
    base = Path(esco_dir)
    result = {}
    for stem in ('occupations_en', 'skills_en', 'occupationSkillRelations_en'):
        paths = list(base.rglob(stem + '.csv'))
        if not paths:
            raise FileNotFoundError(f'Expected ESCO v1.2.1 file: {stem}.csv')
        with paths[0].open(encoding='utf-8-sig', newline='') as handle:
            result[stem + '_rows'] = sum(1 for _ in csv.DictReader(handle))
    result['esco_checked'] = True
    result['esco_directory'] = str(base)
    return result


def make_market(profiles, n, seed, k=30, capacity_mean=3.0, tau=0.1,
                noise_std=0.12, required_skills=2, minimum_ratio=0.45, shift=0.0, utility_mode='structural'):
    rng = np.random.default_rng(seed)
    m = n // 5
    seeker_codes = rng.integers(len(profiles), size=n)
    job_codes = rng.integers(len(profiles), size=m)
    workers = np.clip(profiles[seeker_codes] + rng.normal(0, noise_std, (n, k)), 0, 1).astype(np.float32)
    jobs = profiles[job_codes].astype(np.float32).copy()
    if shift:
        jobs = np.clip(jobs * (1 + shift) + rng.normal(0, 0.03 * shift, jobs.shape), 0, 1)
    capacities = np.clip(rng.poisson(max(0.1, capacity_mean * (1 - shift)), m), 1, 5).astype(np.int32)
    workers_unit = workers / np.maximum(np.linalg.norm(workers, axis=1, keepdims=True), 1e-8)
    jobs_unit = jobs / np.maximum(np.linalg.norm(jobs, axis=1, keepdims=True), 1e-8)
    similarity = workers_unit @ jobs_unit.T
    if utility_mode == 'independent':
        utility = rng.uniform(0, 1, size=similarity.shape)
    elif utility_mode == 'structural':
        utility = similarity
    else:
        raise ValueError('utility_mode must be structural or independent')
    # Qualification is based on the most important job-side skills.
    mandatory = np.argsort(-jobs, axis=1)[:, :required_skills]
    requirements = jobs[np.arange(m)[:, None], mandatory]
    available = workers[:, mandatory]
    eligible = np.all(available >= minimum_ratio * requirements[None, :, :], axis=2)
    candidates = eligible & (similarity >= tau)
    worker_idx, job_idx = np.where(candidates)
    return {'n': int(n), 'm': int(m), 'k': int(k), 'seed': int(seed),
            'workers': workers, 'jobs': jobs, 'capacities': capacities,
            'worker_index': worker_idx.astype(np.int32), 'job_index': job_idx.astype(np.int32),
            'utility': utility[worker_idx, job_idx].astype(np.float64),
            'eligible': candidates, 'total_capacity': int(capacities.sum()),
            'B': int(min(n, capacities.sum())), 'utility_mode': utility_mode}


def save_market(market, file):
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in market.items() if k != 'eligible'})


def save_metadata(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True, default=str), encoding='utf-8')
