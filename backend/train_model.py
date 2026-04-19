"""
Train IPL Win Probability models using synthetic match-state data.
Produces:
  - model.pkl          (2nd innings chase model)
  - model_innings1.pkl (1st innings batting model)
  - encoder.pkl        (shared reference encoder)

2nd innings features:
  Categorical: batting_team, bowling_team, venue
  Numerical:   runs_left, balls_left, wickets_left,
               current_run_rate, required_run_rate

1st innings features:
  Categorical: batting_team, bowling_team, venue
  Numerical:   current_score, balls_left, wickets_left,
               current_run_rate
"""
"""
Train IPL Win Probability models using synthetic match-state data.
Produces:
    - model.pkl (2nd innings chase model)
    - model_innings1.pkl (1st innings batting model)
    - encoder.pkl (shared reference encoder)

2nd innings features:
  Categorical: batting_team, bowling_team, venue, phase
  Numerical: runs_left, balls_left, wickets_left,
               current_run_rate, required_run_rate, run_rate_diff

1st innings features:
  Categorical: batting_team, bowling_team, venue, phase
  Numerical: current_score, balls_left, wickets_left,
               current_run_rate
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from lightgbm import LGBMClassifier

# ── IPL teams and venues ────────────────────────────────────────────
TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. A. Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi Intl. Cricket Stadium, Hyderabad",
    "M. Chinnaswamy Stadium, Bengaluru",
    "Narendra Modi Stadium, Ahmedabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association Stadium, Mohali",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
]

np.random.seed(42)
N = 20_000 # synthetic samples per innings type

# ── 2nd Innings Data ────────────────────────────────────────────────
def generate_innings2_data(n: int) -> pd.DataFrame:
    """Create synthetic second-innings (chase) match situations."""
    rows = []
    for _ in range(n):
        batting_team = np.random.choice(TEAMS)
        bowling_team = np.random.choice([t for t in TEAMS if t!= batting_team])
        venue = np.random.choice(VENUES)

        target = np.random.randint(120, 230)
        overs = round(np.random.uniform(0.1, 19.5), 1)
        overs = min(overs, 19.5)
        balls_bowled = int(overs) * 6 + int((overs % 1) * 10)
        balls_bowled = min(balls_bowled, 119)
        balls_left = 120 - balls_bowled

        wickets = np.random.randint(0, 10)
        wickets_left = 10 - wickets

        max_possible = min(target + 30, int(overs * 12))
        current_score = np.random.randint(0, max(max_possible, 1) + 1)

        runs_left = target - current_score
        crr = current_score / overs if overs > 0 else 0.0
        rrr = (runs_left / (balls_left / 6)) if balls_left > 0 else 99.0

        # New features
        overs_bowled = overs
        phase = 'powerplay' if overs_bowled <= 6 else 'middle' if overs_bowled <= 15 else 'death'
        run_rate_diff = crr - rrr

        # Heuristic win probability
        p_win = 0.5
        if rrr > 0:
            p_win += 0.15 * (crr - rrr) / max(rrr, 1)
        p_win += 0.03 * (wickets_left - 5)
        p_win -= 0.002 * max(runs_left, 0)
        p_win += 0.001 * balls_left
        p_win = np.clip(p_win, 0.02, 0.98)
        result = int(np.random.random() < p_win)

        rows.append({
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "phase": phase,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "current_run_rate": round(crr, 2),
            "required_run_rate": round(rrr, 2),
            "run_rate_diff": round(run_rate_diff, 2),
            "result": result,
        })
    return pd.DataFrame(rows)

# ── 1st Innings Data ────────────────────────────────────────────────
def generate_innings1_data(n: int) -> pd.DataFrame:
    """Create synthetic first-innings batting situations."""
    rows = []
    for _ in range(n):
        batting_team = np.random.choice(TEAMS)
        bowling_team = np.random.choice([t for t in TEAMS if t!= batting_team])
        venue = np.random.choice(VENUES)

        overs = round(np.random.uniform(0.1, 19.5), 1)
        overs = min(overs, 19.5)
        balls_bowled = int(overs) * 6 + int((overs % 1) * 10)
        balls_bowled = min(balls_bowled, 119)
        balls_left = 120 - balls_bowled

        wickets = np.random.randint(0, 10)
        wickets_left = 10 - wickets

        # Score roughly proportional to overs bowled
        max_score = int(overs * 12)
        current_score = np.random.randint(0, max(max_score, 1) + 1)

        crr = current_score / overs if overs > 0 else 0.0

        # New feature
        phase = 'powerplay' if overs <= 6 else 'middle' if overs <= 15 else 'death'

        # Projected total based on CRR and remaining resources
        resource_factor = wickets_left / 10.0
        projected_total = current_score + (crr * (balls_left / 6) * resource_factor)

        # Average IPL total updated for modern era ~183
        avg_total = 183
        p_win = 0.5
        p_win += 0.008 * (projected_total - avg_total)
        p_win += 0.02 * (wickets_left - 5)
        p_win += 0.001 * current_score / max(overs, 0.1) * 0.5
        p_win = np.clip(p_win, 0.05, 0.95)
        result = int(np.random.random() < p_win)

        rows.append({
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "phase": phase,
            "current_score": current_score,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "current_run_rate": round(crr, 2),
            "result": result,
        })
    return pd.DataFrame(rows)

def main():
    # ── Train 2nd Innings Model ──────────────────────────────────────
    print("=" * 50)
    print("Training 2nd Innings Model (chase)")
    print("=" * 50)
    df2 = generate_innings2_data(N)
    X2 = df2.drop(columns=["result"])
    y2 = df2["result"]

    cat_features = ["batting_team", "bowling_team", "venue", "phase"]

    encoder2 = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features),
        ],
        remainder="passthrough",
    )

    pipeline2 = Pipeline([
        ("encoder", encoder2),
        ("clf", CalibratedClassifierCV(
            LGBMClassifier(n_estimators=200, max_depth=4, random_state=42),
            method='isotonic', cv=3
        )),
    ])

    pipeline2.fit(X2, y2)
    proba2 = pipeline2.predict_proba(X2)[:, 1]
    print(f" Train AUC: {roc_auc_score(y2, proba2):.4f}")
    print(f" Train Brier: {brier_score_loss(y2, proba2):.4f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(pipeline2, f)
    print(" Saved model.pkl")

    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder2, f)
    print(" Saved encoder.pkl")

    # ── Train 1st Innings Model ──────────────────────────────────────
    print()
    print("=" * 50)
    print("Training 1st Innings Model (batting first)")
    print("=" * 50)
    df1 = generate_innings1_data(N)
    X1 = df1.drop(columns=["result"])
    y1 = df1["result"]

    encoder1 = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features),
        ],
        remainder="passthrough",
    )

    pipeline1 = Pipeline([
        ("encoder", encoder1),
        ("clf", CalibratedClassifierCV(
            LGBMClassifier(n_estimators=200, max_depth=4, random_state=42),
            method='isotonic', cv=3
        )),
    ])

    pipeline1.fit(X1, y1)
    proba1 = pipeline1.predict_proba(X1)[:, 1]
    print(f" Train AUC: {roc_auc_score(y1, proba1):.4f}")
    print(f" Train Brier: {brier_score_loss(y1, proba1):.4f}")

    with open("model_innings1.pkl", "wb") as f:
        pickle.dump(pipeline1, f)
    print(" Saved model_innings1.pkl")

    # ── Sanity checks ────────────────────────────────────────────────
    print()
    print("Sanity checks:")
    prob2 = pipeline2.predict_proba(X2.iloc[:1])[0]
    print(f" 2nd innings sample: lose={prob2[0]:.3f} win={prob2[1]:.3f}")
    prob1 = pipeline1.predict_proba(X1.iloc[:1])[0]
    print(f" 1st innings sample: lose={prob1[0]:.3f} win={prob1[1]:.3f}")

if __name__ == "__main__":
    main()
"""import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# ── IPL teams and venues ────────────────────────────────────────────
TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. A. Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi Intl. Cricket Stadium, Hyderabad",
    "M. Chinnaswamy Stadium, Bengaluru",
    "Narendra Modi Stadium, Ahmedabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association Stadium, Mohali",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
]

np.random.seed(42)
N = 20_000  # synthetic samples per innings type


# ── 2nd Innings Data ────────────────────────────────────────────────
def generate_innings2_data(n: int) -> pd.DataFrame:
    """Create synthetic second-innings (chase) match situations."""
    rows = []
    for _ in range(n):
        batting_team = np.random.choice(TEAMS)
        bowling_team = np.random.choice([t for t in TEAMS if t != batting_team])
        venue = np.random.choice(VENUES)

        target = np.random.randint(120, 230)
        overs = round(np.random.uniform(0.1, 19.5), 1)
        overs = min(overs, 19.5)
        balls_bowled = int(overs) * 6 + int((overs % 1) * 10)
        balls_bowled = min(balls_bowled, 119)
        balls_left = 120 - balls_bowled

        wickets = np.random.randint(0, 10)
        wickets_left = 10 - wickets

        max_possible = min(target + 30, int(overs * 12))
        current_score = np.random.randint(0, max(max_possible, 1) + 1)

        runs_left = target - current_score
        crr = current_score / overs if overs > 0 else 0.0
        rrr = (runs_left / (balls_left / 6)) if balls_left > 0 else 99.0

        # Heuristic win probability
        p_win = 0.5
        if rrr > 0:
            p_win += 0.15 * (crr - rrr) / max(rrr, 1)
        p_win += 0.03 * (wickets_left - 5)
        p_win -= 0.002 * max(runs_left, 0)
        p_win += 0.001 * balls_left
        p_win = np.clip(p_win, 0.02, 0.98)
        result = int(np.random.random() < p_win)

        rows.append({
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "current_run_rate": round(crr, 2),
            "required_run_rate": round(rrr, 2),
            "result": result,
        })
    return pd.DataFrame(rows)


# ── 1st Innings Data ────────────────────────────────────────────────
def generate_innings1_data(n: int) -> pd.DataFrame:
    """Create synthetic first-innings batting situations."""
    rows = []
    for _ in range(n):
        batting_team = np.random.choice(TEAMS)
        bowling_team = np.random.choice([t for t in TEAMS if t != batting_team])
        venue = np.random.choice(VENUES)

        overs = round(np.random.uniform(0.1, 19.5), 1)
        overs = min(overs, 19.5)
        balls_bowled = int(overs) * 6 + int((overs % 1) * 10)
        balls_bowled = min(balls_bowled, 119)
        balls_left = 120 - balls_bowled

        wickets = np.random.randint(0, 10)
        wickets_left = 10 - wickets

        # Score roughly proportional to overs bowled
        max_score = int(overs * 12)
        current_score = np.random.randint(0, max(max_score, 1) + 1)

        crr = current_score / overs if overs > 0 else 0.0

        # Projected total based on CRR and remaining resources
        resource_factor = wickets_left / 10.0
        projected_total = current_score + (crr * (balls_left / 6) * resource_factor)

        # Average IPL total ~165. Higher projected = more likely to win.
        avg_total = 165
        p_win = 0.5
        p_win += 0.008 * (projected_total - avg_total)
        p_win += 0.02 * (wickets_left - 5)
        p_win += 0.001 * current_score / max(overs, 0.1) * 0.5
        p_win = np.clip(p_win, 0.05, 0.95)
        result = int(np.random.random() < p_win)

        rows.append({
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "current_score": current_score,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "current_run_rate": round(crr, 2),
            "result": result,
        })
    return pd.DataFrame(rows)


def main():
    # ── Train 2nd Innings Model ──────────────────────────────────────
    print("=" * 50)
    print("Training 2nd Innings Model (chase)")
    print("=" * 50)
    df2 = generate_innings2_data(N)
    X2 = df2.drop(columns=["result"])
    y2 = df2["result"]

    cat_features = ["batting_team", "bowling_team", "venue"]

    encoder2 = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features),
        ],
        remainder="passthrough",
    )

    pipeline2 = Pipeline([
        ("encoder", encoder2),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])

    pipeline2.fit(X2, y2)
    print(f"  Training accuracy: {pipeline2.score(X2, y2):.4f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(pipeline2, f)
    print("  Saved model.pkl")

    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder2, f)
    print("  Saved encoder.pkl")

    # ── Train 1st Innings Model ──────────────────────────────────────
    print()
    print("=" * 50)
    print("Training 1st Innings Model (batting first)")
    print("=" * 50)
    df1 = generate_innings1_data(N)
    X1 = df1.drop(columns=["result"])
    y1 = df1["result"]

    encoder1 = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features),
        ],
        remainder="passthrough",
    )

    pipeline1 = Pipeline([
        ("encoder", encoder1),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])

    pipeline1.fit(X1, y1)
    print(f"  Training accuracy: {pipeline1.score(X1, y1):.4f}")

    with open("model_innings1.pkl", "wb") as f:
        pickle.dump(pipeline1, f)
    print("  Saved model_innings1.pkl")

    # ── Sanity checks ────────────────────────────────────────────────
    print()
    print("Sanity checks:")
    prob2 = pipeline2.predict_proba(X2.iloc[:1])[0]
    print(f"  2nd innings sample: lose={prob2[0]:.3f}  win={prob2[1]:.3f}")
    prob1 = pipeline1.predict_proba(X1.iloc[:1])[0]
    print(f"  1st innings sample: lose={prob1[0]:.3f}  win={prob1[1]:.3f}")


if __name__ == "__main__":
    main()"""
