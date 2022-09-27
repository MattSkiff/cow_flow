import os
import 

# want to plot width of prediction interval from NF (as measure of uncertainty)
# against actual error from baseline model
# high correspondence shows the NF is correctly assigning uncertainty to predictions

# want to combine errors across all predictions from all models
    
error_file_path = os.path.join(g.VIZ_DATA_DIR,error_file_name)
prediction_interval_file_path = os.path.join(g.VIZ_DATA_DIR,prediction_interval_file_name)

errors = []; prediction_intervals = []

with open(error_file_path) as f:
    errors_input = f.readlines()

    for line in errors_input:
        errors.append(line)    

with open(prediction_interval_file_path) as f:
    prediction_interval_input = f.readlines()

    for line in prediction_interval_input:
        line = np.array(line[1:-2].split(),dtype=np.float32)
        prediction_intervals.append(line)  
        
intervals_df = pd.DataFrame(np.array(prediction_intervals), columns=['lb', 'ub'])
pred_width = np.array(np.abs(intervals_df['lb']-intervals_df['ub']))
errors = np.array(errors,dtype=np.float32)

fig, ax = plt.subplots(1)
#ax.vlines(errors,intervals_df['lb'],intervals_df['ub'])
ax.scatter(errors,pred_width)
ax.set_xlabel('Baseline LCFCN error size'.format(Path(error_file_path).stem))
ax.set_ylabel('NF prediction interval width')

plt.show()