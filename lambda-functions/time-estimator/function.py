import json
import math
import random
import statistics
from collections import defaultdict

import boto3
import requests

from rating import RatingSystem, ContestType
from stats import single_regression, safe_log, safe_sigmoid, minimize, normal_pdf, normal_cdf


def fit_2plm_irt(xs, ys):
    def f(sample, args):
        x, y = sample
        a, b = args
        p = safe_sigmoid(a * x + b)
        return -safe_log(p if y == 1. else (1 - p))

    def grad_f(sample, args):
        x, y = sample
        a, b = args
        p = safe_sigmoid(a * x + b)
        grad_a = x * (p - y)
        grad_b = (p - y)
        return [-grad_a, -grad_b]

    x_scale = 1000.
    scxs = [x / x_scale for x in xs]
    samples = list(zip(scxs, ys))

    best_args, _ = minimize([0., 0.], samples, f, grad_f, 100000, 1., 20191019)
    a, b = best_args

    a /= x_scale
    return -b / a, -a


def evaluate_2plm_irt(xs, ys, difficulty, discrimination):
    n = len(xs)
    if difficulty is None or discrimination is None:
        logl = n * math.log(0.5)
    else:
        logl = 0
        for x, y in zip(xs, ys):
            p = safe_sigmoid(-discrimination * (x - difficulty))
            logl += safe_log(p if y == 1. else (1 - p))
    return logl, n


def inverse_adjust_rating(rating, prev_contests):
    if rating <= 0:
        return float("nan")
    if rating <= 400:
        rating = 400 * (1 - math.log(400 / rating))
    adjustment = (math.sqrt(1 - (0.9 ** (2 * prev_contests))) /
                  (1 - 0.9 ** prev_contests) - 1) / (math.sqrt(19) - 1) * 1200
    return rating + adjustment


def is_very_easy_problem(task_screen_name):
    return task_screen_name.startswith("abc") and task_screen_name[-1] in {"a", "b"} and int(
        task_screen_name[3:6]) >= 42


def fit_time_model(raw_ratings, time_log_secs, acs):
    ac_ratings, ac_time_logs = [], []
    wa_ratings, wa_time_logs = [], []
    for i in range(len(acs)):
        if acs[i] == 1.:
            ac_ratings.append(raw_ratings[i])
            ac_time_logs.append(time_log_secs[i])
        else:
            wa_ratings.append(raw_ratings[i])
            wa_time_logs.append(time_log_secs[i])
    n_ac, n_wa = len(ac_ratings), len(wa_ratings)

    slope, intercept = single_regression(ac_ratings, ac_time_logs)
    variance = statistics.variance([slope * ac_ratings[i] + intercept - ac_time_logs[i] for i in range(n_ac)])

    def f(sample, args):
        x, y, t = sample
        a, b, vl = args
        v = math.exp(vl)
        sd = math.sqrt(v)
        pred_y = a * x + b
        if t == 1.:
            # ac
            return -safe_log(normal_pdf(y - pred_y, 0., sd))
        else:
            # wa
            return -safe_log(1. - normal_cdf(y, pred_y, sd))

    def grad_f(sample, args):
        grads = []
        delta = 1e-8
        for i in range(len(args)):
            new_args = list(args)
            new_args[i] = args[i] + delta
            l = f(sample, new_args)
            new_args[i] = args[i] - delta
            r = f(sample, new_args)
            grads.append((l - r) / (2 * delta))
        return grads

    rating_scale = 1000.
    scaled_ratings = [r / rating_scale for r in raw_ratings]
    samples = list(zip(scaled_ratings, time_log_secs, acs))

    # parameter transform
    # slope -> slope * rating_scale: to improve gradient descent based optimization
    # variance -> log(variance): to eliminate variance > 0 constraint (minimize() does not support constraints)
    best_args, _ = minimize([slope * rating_scale, intercept, math.log(variance)], samples, f, grad_f, 20000, 0.001, 20191019)
    scaled_slope, best_intercept, log_variance = best_args
    best_slope = scaled_slope / rating_scale
    best_variance = math.exp(log_variance)
    return best_slope, best_intercept, best_variance


def fit_problem_model(user_results, task_screen_name):
    max_score = max(task_result[task_screen_name + ".score"] for task_result in user_results)
    if max_score == 0.:
        print(f"The problem {task_screen_name} is not solved by any competitors. skipping.")
        return {}
    for task_result in user_results:
        task_result[task_screen_name + ".ac"] = float(task_result[task_screen_name + ".score"] == max_score)
    ac_elapsed = [task_result[task_screen_name + ".elapsed"]
                  for task_result in user_results if task_result[task_screen_name + ".ac"] == 1.]
    first_ac = min(ac_elapsed)

    recurring_users = [task_result for task_result in user_results if
                       task_result["prev_contests"] > 0 and task_result["rating"] > 0]
    for task_result in recurring_users:
        task_result["raw_rating"] = inverse_adjust_rating(task_result["rating"], task_result["prev_contests"])
    time_model_sample_users = []
    for task_result in recurring_users:
        if task_result[task_screen_name + ".ac"] == 1.:
            if task_result[task_screen_name + ".time"] > first_ac / 2:
                time_model_sample_users.append(task_result)
        else:
            time_model_sample_users.append(task_result)

    model = {}
    if len(time_model_sample_users) < 40:
        print(
            f"{task_screen_name}: insufficient data ({len(time_model_sample_users)} users). skip estimating time model.")
    elif sum(task_result[task_screen_name + ".ac"] for task_result in time_model_sample_users) < 5:
        print(
            f"{task_screen_name}: insufficient accepted data. skip estimating time model.")
    else:
        raw_ratings = [task_result["raw_rating"]
                       for task_result in time_model_sample_users]
        time_secs = [task_result[task_screen_name + ".time"] /
                     (10 ** 9) for task_result in time_model_sample_users]
        time_logs = [math.log(t) for t in time_secs]
        acs = [task_result[task_screen_name + ".ac"] for task_result in time_model_sample_users]
        slope, intercept, variance = fit_time_model(raw_ratings, time_logs, acs)
        print(
            f"{task_screen_name}: time [sec] = exp({slope} * raw_rating + {intercept})")
        if slope > 0:
            print("slope is positive. ignoring unreliable estimation.")
        else:
            model["slope"] = slope
            model["intercept"] = intercept
            model["variance"] = variance

    if is_very_easy_problem(task_screen_name):
        # ad-hoc. excluding high-rating competitors from abc-a/abc-b dataset. They often skip these problems.
        difficulty_dataset = [task_result for task_result in recurring_users if task_result["is_rated"]]
    else:
        difficulty_dataset = recurring_users
    if len(difficulty_dataset) < 40:
        print(
            f"{task_screen_name}: insufficient data ({len(difficulty_dataset)} users). skip estimating difficulty model.")
    elif all(task_result[task_screen_name + ".ac"] for task_result in difficulty_dataset):
        print("all contestants got AC. skip estimating difficulty model.")
    elif not any(task_result[task_screen_name + ".ac"] for task_result in difficulty_dataset):
        print("no contestants got AC. skip estimating difficulty model.")
    else:
        d_raw_ratings = [task_result["raw_rating"]
                         for task_result in difficulty_dataset]
        d_accepteds = [task_result[task_screen_name + ".ac"]
                       for task_result in difficulty_dataset]
        difficulty, discrimination = fit_2plm_irt(
            d_raw_ratings, d_accepteds)
        print(
            f"difficulty: {difficulty}, discrimination: {discrimination}")
        if discrimination < 0:
            print("discrimination is negative. ignoring unreliable estimation.")
        elif difficulty > 6000:
            print("extreme difficulty. rejecting this estimation.")
        else:
            model["difficulty"] = difficulty
            model["discrimination"] = discrimination
        loglikelihood, users = evaluate_2plm_irt(d_raw_ratings, d_accepteds, difficulty, discrimination)
        model["irt_loglikelihood"] = loglikelihood
        model["irt_users"] = users
    return model


def fetch_dataset_for_contest(contest_name, existing_problem, duration_second):
    try:
        results = requests.get(
            "https://atcoder.jp/contests/{}/standings/json".format(contest_name)).json()
    except json.JSONDecodeError as e:
        print(f"{e}")
        return {}
    task_names = {task["TaskScreenName"]: task["TaskName"]
                  for task in results["TaskInfo"]}

    user_results = []
    standings_data = results["StandingsData"]
    standings_data.sort(key=lambda result_row: result_row["Rank"])
    standings = []
    for result_row in standings_data:
        total_submissions = result_row["TotalResult"]["Count"]
        if total_submissions == 0:
            continue

        is_rated = result_row["IsRated"]
        rating = result_row["OldRating"]
        prev_contests = result_row["Competitions"]
        user_name = result_row["UserScreenName"]

        standings.append(user_name)
        user_row = {
            "is_rated": is_rated,
            "rating": rating,
            "prev_contests": prev_contests,
            "user_name": user_name
        }
        prev_accepted_times = [0] + [task_result["Elapsed"]
                                     for task_result in result_row["TaskResults"].values() if task_result["Score"] > 0]
        default_time = duration_second * (10 ** 9) - max(prev_accepted_times)
        for task_name in task_names:
            user_row[task_name + ".score"] = 0.
            user_row[task_name + ".time"] = default_time
            user_row[task_name + ".elapsed"] = duration_second * (10 ** 9)

        user_row["last_ac"] = max(prev_accepted_times)
        for task_screen_name, task_result in result_row["TaskResults"].items():
            user_row[task_screen_name + ".score"] = task_result["Score"]
            if task_result["Score"] > 0:
                elapsed = task_result["Elapsed"]
                penalty = task_result["Penalty"] * 5 * 60 * (10 ** 9)
                user_row[task_screen_name + ".elapsed"] = elapsed
                user_row[task_screen_name + ".time"] = penalty + elapsed - \
                                                       max(t for t in prev_accepted_times if t < elapsed)
        user_results.append(user_row)

    if len(user_results) == 0:
        print(
            f"There are no participants/submissions for contest {contest_name}. Ignoring.")
        return {}

    user_results_by_problem = defaultdict(list)
    for task_screen_name in task_names.keys():
        if task_screen_name in existing_problem:
            print(f"The problem model for {task_screen_name} already exists. skipping.")
            continue
        user_results_by_problem[task_screen_name] += user_results
    return user_results_by_problem, standings


def get_current_models():
    try:
        return requests.get("https://kenkoooo.com/atcoder/resources/problem-models.json").json()
    except Exception as e:
        print(f"Failed to fetch existing models.\n{e}")
        return {}


def infer_contest_type(contest) -> ContestType:
    if contest["rate_change"] == "All":
        return ContestType.AGC
    elif contest["rate_change"] == " ~ 2799":
        return ContestType.NEW_ARC
    elif contest["rate_change"] == " ~ 1999":
        return ContestType.NEW_ABC
    elif contest["rate_change"] == " ~ 1199":
        return ContestType.OLD_ABC
    # rate_change == "-"
    elif contest["id"].startswith("arc"):
        return ContestType.OLD_UNRATED_ARC
    elif contest["id"].startswith("abc"):
        return ContestType.OLD_UNRATED_ABC
    else:
        return ContestType.UNRATED


def all_rated_contests():
    # Gets all contest IDs and their contest type
    # The result is ordered by the start time.
    contests = requests.get(
        "https://kenkoooo.com/atcoder/resources/contests.json").json()
    contests.sort(key=lambda contest: contest["start_epoch_second"])
    contests_and_types = [(contest, infer_contest_type(contest)) for contest in contests]
    return [(contest, contest_type) for contest, contest_type in contests_and_types if
            contest_type != ContestType.UNRATED]


def run(target, overwrite):
    recompute_history = target is None and overwrite
    rated_contests = all_rated_contests()
    if target is None:
        target = rated_contests
    else:
        target = [(contest, contest_type) for contest, contest_type in rated_contests if contest["id"] in target]
    current_models = get_current_models()
    existing_problems = current_models.keys() if not overwrite else set()

    print(f"Fetching dataset from {len(target)} contests.")
    dataset_by_problem = defaultdict(list)
    rating_system = RatingSystem()
    competition_history_by_id = defaultdict(set)
    experimental_problems = set()
    for contest, contest_type in target:
        contest_id = contest["id"]
        duration_second = contest["duration_second"]
        is_old_contest = not contest_type.is_rated
        user_results_by_problem, standings = fetch_dataset_for_contest(contest_id, existing_problems, duration_second)
        for problem, data_points in user_results_by_problem.items():
            if recompute_history:
                # overwrite competition history, and rating if necessary
                if is_old_contest:
                    # contests before official rating system. using the emulated rating
                    experimental_problems.add(problem)
                    for data_point in data_points:
                        prev_contests = rating_system.competition_count(data_point["user_name"])
                        data_point["prev_contests"] = prev_contests
                        data_point["rating"] = rating_system.calc_rating(
                            data_point["user_name"]) if prev_contests > 0 else 0
                else:
                    # contests after official rating system. using the official rating
                    for data_point in data_points:
                        competition_history_by_id[data_point["user_name"]].add(contest_id)
                    for data_point in data_points:
                        data_point["prev_contests"] = len(competition_history_by_id[data_point["user_name"]]) - 1
            dataset_by_problem[problem] += data_points
        if recompute_history and is_old_contest:
            print("Updating user rating with the result of {}".format(contest_id))
            rating_system.update(standings, contest_type)
    print(f"Estimating time models of {len(target)} contests.")
    results = current_models
    for problem, data_points in dataset_by_problem.items():
        model = fit_problem_model(data_points, problem)
        model["is_experimental"] = problem in experimental_problems
        results[problem] = model
    return results


def handler(event, context):
    target = event.get("target")
    overwrite = event.get("overwrite", False)
    bucket = event.get("bucket", "kenkoooo.com")
    object_key = event.get("object_key", "resources/problem-models.json")

    results = run(target, overwrite)
    print("Estimation completed. Saving results in S3")
    s3 = boto3.resource('s3')
    s3.Object(bucket, object_key).put(Body=json.dumps(
        results), ContentType="application/json")
