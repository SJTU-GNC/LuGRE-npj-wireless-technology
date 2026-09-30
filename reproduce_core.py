"""Reproduce the current manuscript from released, version-locked derived inputs.

No raw public files are downloaded and no original artifacts are overwritten.
Only load the supplied trusted joblib file after checking its SHA256 manifest.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
import joblib
from scipy.signal import savgol_filter
from sklearn.base import clone

KEYS=['minute_utc','op','signal_name','svid']
CORE=Path('table/algorithm/cn0_trend_residual_tuned_no_leakage')

def smooth(frame, values, isolate_splits):
    """The original nine-point median + degree-2 Savitzky-Golay output head."""
    work=frame.copy(); work['_value']=values
    out=pd.Series(np.nan,index=frame.index,dtype=float)
    cols=['op','signal_name','svid']
    if isolate_splits:cols=['evaluation_split']+cols
    for _,group in work.groupby(cols,dropna=False):
        ordered=group.sort_values('minute_utc')
        segments=ordered['minute_utc'].diff().gt(pd.Timedelta(minutes=1.5)).cumsum()
        for _,part in ordered.groupby(segments):
            v=pd.to_numeric(part['_value'],errors='coerce').interpolate(limit_direction='both')
            if not v.notna().any():continue
            v=v.rolling(9,center=True,min_periods=1).median()
            if len(v)>=9:v=pd.Series(savgol_filter(v.to_numpy(float),9,2,mode='interp'),index=v.index)
            out.loc[part.index]=v.to_numpy(float)
    return out.to_numpy(float)

def metric(y,p):
    e=np.asarray(p)-np.asarray(y)
    return {'n':len(e),'rmse_dbhz':float(np.sqrt(np.mean(e**2))),
            'mae_dbhz':float(np.mean(np.abs(e))),'bias_dbhz':float(np.mean(e))}

def fit_stage(frame, artifact, splits):
    sel=frame['evaluation_split'].isin(splits)&frame['trend_training_eligible']&frame['residual_trend_target_db'].notna()
    part=frame.loc[sel]
    beta=part.groupby('signal_name')['residual_trend_target_db'].median().to_dict()
    model=clone(artifact['model'])
    y=part['residual_trend_target_db']-part['signal_name'].map(beta)
    model.fit(part[artifact['model_features']],y)
    return model,beta,int(sel.sum())

def predict(frame,model,beta,features):
    return np.asarray(model.predict(frame[features]),float)+frame['signal_name'].map(beta).to_numpy(float)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'data')
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'run_output')
    ap.add_argument('--refit',action='store_true',help='Also refit train-only and train+validation models (no new selection).')
    args=ap.parse_args()
    model_path=args.data/CORE/'cn0_trend_residual_tuned_no_leakage.joblib'
    expected='4ce4d4fd0344d1b500e182cb1e21852335e1a68cd856e20a6a86d3d23e8bceb8'
    actual=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError('Model hash does not match the original manuscript branch.')
    artifact=joblib.load(model_path)
    feature_path=args.data/'derived/one_minute_features.csv.gz'
    if feature_path.exists():frame=pd.read_csv(feature_path,parse_dates=['minute_utc'],float_precision='round_trip')
    else:frame=joblib.load(args.data/'table/algorithm/cn0_ai_residual_story/final_split_feature_frame.joblib')
    frame=frame.reset_index(drop=True)
    frame['minute_utc']=pd.to_datetime(frame['minute_utc'],utc=True)
    frame['trend_training_eligible']=frame['trend_training_eligible'].astype(bool)
    saved=pd.read_csv(args.data/CORE/'cn0_trend_residual_tuned_no_leakage_predictions.csv',parse_dates=['minute_utc'],float_precision='round_trip')
    saved['minute_utc']=pd.to_datetime(saved['minute_utc'],utc=True)
    frame['svid']=frame['svid'].astype(str)
    saved['svid']=saved['svid'].astype(str)
    # Verify the saved cache and prediction tables contain identical ordered keys.
    if frame[KEYS].duplicated().any() or saved[KEYS].duplicated().any():raise RuntimeError('Duplicate sample key')
    saved=frame[KEYS].merge(saved,on=KEYS,how='left',validate='one_to_one')
    if saved['evaluation_split'].isna().any():raise RuntimeError('Missing saved prediction')
    features=artifact['model_features']
    raw=predict(frame,artifact['model'],artifact['signal_beta_db'],features)
    final_eval=frame['cn0_physics_trend_dbhz'].to_numpy(float)+smooth(frame,raw,True)
    final_display=frame['cn0_physics_trend_dbhz'].to_numpy(float)+smooth(frame,raw,False)
    hold=frame['evaluation_split'].isin(['test','external_holdout']).to_numpy()
    checks={'model_sha256':actual,'feature_count':len(features),'total_minute_samples':len(frame),
            'raw_prediction_max_abs_diff_db':float(np.nanmax(np.abs(raw-saved['ai_trend_residual_raw_pred_db']))),
            'test_prediction_max_abs_diff_dbhz':float(np.nanmax(np.abs(final_eval[hold]-saved.loc[hold,'cn0_physics_ai_eval_dbhz']))),
            'display_prediction_max_abs_diff_dbhz':float(np.nanmax(np.abs(final_display-saved['cn0_physics_ai_trend_dbhz'])))}
    rows=[]
    for split in ['test','external_holdout']:
        use=frame['evaluation_split'].eq(split)&frame['trend_training_eligible']
        for op in ['all']+sorted(frame.loc[use,'op'].unique()):
            take=use if op=='all' else use&frame['op'].eq(op)
            y=frame.loc[take,'cn0_observed_trend_dbhz']
            base=frame.loc[take,'cn0_physics_trend_dbhz']
            beta=frame.loc[take,'signal_name'].map(artifact['signal_beta_db'])
            for name,p in [('physical_baseline',base),('baseline_plus_beta',base+beta),('final_model',final_eval[take])]:
                rows.append({'split':split,'operation':op,'model':name,**metric(y,p)})
    if args.refit:
        fitted,beta,n=fit_stage(frame,artifact,['train','validation'])
        refit_raw=predict(frame,fitted,beta,features)
        checks.update(refit_samples=n,refit_beta_db={k:float(v) for k,v in beta.items()},
                      refit_raw_max_abs_diff_db=float(np.max(np.abs(refit_raw-raw))))
        train_model,train_beta,n_train=fit_stage(frame,artifact,['train'])
        val_pred=frame['cn0_physics_trend_dbhz'].to_numpy(float)+smooth(frame,predict(frame,train_model,train_beta,features),True)
        v=frame['evaluation_split'].eq('validation')&frame['trend_training_eligible']
        checks.update(train_only_samples=n_train,validation_prediction_max_abs_diff_dbhz=float(np.nanmax(np.abs(val_pred[v]-saved.loc[v,'cn0_physics_ai_eval_dbhz']))))
        rows.append({'split':'validation','operation':'all','model':'train_only_selected_model',**metric(frame.loc[v,'cn0_observed_trend_dbhz'],val_pred[v])})
    checks['passed']=all(v<1e-7 for k,v in checks.items() if 'max_abs_diff' in k)
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out/'recomputed_metrics.csv',index=False)
    (args.out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
    print(pd.DataFrame(rows).query("operation=='all'").to_string(index=False))
    if not checks['passed']:sys.exit(2)

if __name__=='__main__':main()
