"""
Summary performance csv files for triton Inductor e2e tests 

precision:  'amp_bf16','amp_fp16','bfloat16','float16','float32'
mode: 'inference','training'

Usage:
  python perf_summary.py -s huggingface -p amp_bf16 amp_fp16
"""
import argparse
from styleframe import StyleFrame
from scipy.stats import gmean
import pandas as pd

parser = argparse.ArgumentParser(description="Generate report")
parser.add_argument('-s','--suite',default='huggingface',choices=["torchbench","huggingface","timm_models"],type=str,help='model suite name')
parser.add_argument('-p','--precision',default='amp_bf16 amp_fp16',nargs='*',type=str,help='precision')
args=parser.parse_args()

passrate_values={}
geomean_values={}

# refer to https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/runner.py#L757-L778
def percentage(part, whole, decimals=2):
    if whole == 0:
        return 0
    return round(100 * float(part) / float(whole), decimals)

def get_passing_entries(df,column_name):
    return df[column_name][df[column_name] > 0]

def caculate_geomean(df,column_name):
    cleaned_df = get_passing_entries(df,column_name).clip(1)
    if cleaned_df.empty:
        return "0.0x"
    return f"{gmean(cleaned_df):.2f}x"

def caculate_passrate(df,compiler):
    total = len(df.index)
    passing = df[df[compiler] > 0.0][compiler].count()
    perc = int(percentage(passing, total, decimals=0))
    return f"{perc}%, {passing}/{total}"
    
def get_perf_csv(precision,mode):
    target_path='inductor_log/huggingface/'+precision+'/inductor_'+args.suite+'_'+precision+'_'+mode+'_xpu_performance.csv'
    target_ori_data=pd.read_csv(target_path,index_col=0)
    target_data=target_ori_data[['name','batch_size','speedup','abs_latency']]
    target_data=target_data.copy()
    target_data.sort_values(by=['name'], key=lambda col: col.str.lower(),inplace=True)
    return target_data

def process(input,precision,mode):
    data_new=input[['name','batch_size','speedup','abs_latency']].rename(columns={'name':'name','batch_size':'batch_size','speedup':'speedup',"abs_latency":'inductor'})
    data_new['inductor']=data_new['inductor'].astype(float).div(1000)
    data_new['speedup']=data_new['speedup'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    data_new['eager'] = data_new['speedup'] * data_new['inductor']
    global geomean_values, passrate_values
    geomean_values[str(precision)+'_'+str(mode)] = caculate_geomean(data_new,'speedup')
    passrate_values[str(precision)+'_'+str(mode)] = caculate_passrate(data_new,'inductor')
    data = StyleFrame({'name': list(data_new['name']),
                'batch_size': list(data_new['batch_size']),
                'speedup': list(data_new['speedup']),
                'inductor': list(data_new['inductor']),
                'eager': list(data_new['eager'])})
    data.set_column_width(1, 10)
    data.set_column_width(2, 18)
    data.set_column_width(3, 18)
    data.set_column_width(4, 18)
    data.set_column_width(5, 15)
    data.set_row_height(rows=data.row_indexes, height=15)
    return data

def update_details(precision,mode,excel):
    h = {"A": 'Model suite',"B": '', "C": "target", "D": '', "E": '',"F": '',"G":''}
    head = StyleFrame(pd.DataFrame(h, index=[0]))
    head.set_column_width(1, 15)
    head.set_row_height(rows=[1], height=15)

    head.to_excel(excel_writer=excel, sheet_name=precision+'_'+mode, index=False,startrow=0,header=False)
    target_raw_data=get_perf_csv(precision,mode)
    target_data=process(target_raw_data,precision,mode)
    target_data.to_excel(excel_writer=excel,sheet_name=precision+'_'+mode,index=False,startrow=1,startcol=1)

def update_summary(excel):
    data = {
        'Test Secnario':['AMP_BF16 Inference', ' ','AMP_BF16 Training',' ','AMP_FP16 Inference', ' ','AMP_FP16 Training', ' ','BF16 Inference', ' ','BF16 Training', ' ','FP16 Inference', ' ','FP16 Training', ' ','FP32 Inference', ' ','FP32 Training', ' '],
        'Comp Item':['Pass Rate', 'Geomean Speedup','Pass Rate','Geomean Speedup','Pass Rate', 'Geomean Speedup','Pass Rate','Geomean Speedup','Pass Rate', 'Geomean Speedup','Pass Rate','Geomean Speedup','Pass Rate', 'Geomean Speedup','Pass Rate','Geomean Speedup','Pass Rate', 'Geomean Speedup','Pass Rate','Geomean Speedup'],
        'Date':[' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' '],
        'Compiler':['inductor', 'inductor', 'inductor', 'inductor','inductor', 'inductor', 'inductor', 'inductor','inductor', 'inductor', 'inductor', 'inductor','inductor', 'inductor', 'inductor', 'inductor','inductor', 'inductor', 'inductor', 'inductor'],
        'torchbench':[' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' '],
        'huggingface':[' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' '],
        'timm_models ':[' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' ',' ', ' ', ' ', ' ']
    }
    summary=pd.DataFrame(data)
    # passrate
    summary.iloc[0:1,5:6]= passrate_values['amp_bf16_inference']
    summary.iloc[2:3,5:6]= passrate_values['amp_bf16_training']
    summary.iloc[4:5,5:6]= passrate_values['amp_fp16_inference']
    summary.iloc[6:7,5:6]= passrate_values['amp_fp16_training']
    # geomean_speedup
    summary.iloc[1:2,5:6]= geomean_values['amp_bf16_inference']
    summary.iloc[3:4,5:6]= geomean_values['amp_bf16_training']
    summary.iloc[5:6,5:6]= geomean_values['amp_fp16_inference']
    summary.iloc[7:8,5:6]= geomean_values['amp_fp16_training']

    if 'bfloat16' in args.precision:
        summary.iloc[8:9,5:6]= passrate_values['bfloat16_inference']
        summary.iloc[10:11,5:6]= passrate_values['bfloat16_training']
        summary.iloc[9:10,5:6]= geomean_values['bfloat16_inference']
        summary.iloc[11:12,5:6]= geomean_values['bfloat16_training']
    if 'float16' in args.precision:
        summary.iloc[12:13,5:6]= passrate_values['float16_inference']
        summary.iloc[14:15,5:6]= passrate_values['float16_training']
        summary.iloc[13:14,5:6]= geomean_values['float16_inference']
        summary.iloc[15:16,5:6]= geomean_values['float16_training']    
    if 'float32' in args.precision:
        summary.iloc[16:17,5:6]= passrate_values['float32_inference']
        summary.iloc[18:19,5:6]= passrate_values['float32_training']
        summary.iloc[17:18,5:6]= geomean_values['float32_inference']
        summary.iloc[19:20,5:6]= geomean_values['float32_training']

    print("===============================================Summary========================================")
    print(summary)
    print("===============================================Summary========================================")

    sf = StyleFrame(summary)
    for i in range(1,8):
        sf.set_column_width(i, 18)
    for j in range(1,22):
        sf.set_row_height(j, 30)
    sf.to_excel(sheet_name='Summary',excel_writer=excel)

def generate_report(excel,precision_list):
    for p in precision_list:
        for m in ['inference','training']:
            update_details(p,m,excel)
    update_summary(excel)

def excel_postprocess(file,precison):
    wb=file.book
    # Summary
    ws=wb['Summary']
    for i in range(2,21,2):
        ws.merge_cells(start_row=i,end_row=i+1,start_column=1,end_column=1)
    # details
    for p in precison:
        for m in ['inference','training']:
            wdt=wb[p+'_'+m]
            wdt.merge_cells(start_row=1,end_row=2,start_column=1,end_column=1)
            wdt.merge_cells(start_row=1,end_row=1,start_column=3,end_column=6)
    wb.save(file)

if __name__ == '__main__':
    excel = StyleFrame.ExcelWriter('inductor_log/'+str(args.suite)+'/Inductor_'+args.suite+'_E2E_Test_Report.xlsx')
    generate_report(excel,args.precision)
    excel_postprocess(excel,args.precision)
