print(type(trained_model))
# For sklearn Pipelines
try: 
    import sklearn
    from sklearn.pipeline import Pipeline
    print("is Pipeline:", isinstance(trained_model, Pipeline))
except Exception: 
    pass

# For LightGBM sklearn wrapper
try:
    import lightgbm as lgb
    from lightgbm.sklearn import LGBMClassifier
    print("is LGBMClassifier:", isinstance(trained_model, LGBMClassifier))
except Exception:
    pass

# For raw LightGBM Booster (not sklearn)
try:
    import lightgbm as lgb
    print("is Booster:", isinstance(trained_model, lgb.Booster))
except Exception:
    pass
