import glob, os, csv
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = "/home/andreasabo/Documents/HNProject"

def extract_label_csvs_to_df(csv_list):
	'''
	(list of paths to label csvs) -> data_frame
	'''
	all_labels = pd.DataFrame()

	header_names = ['num_in_seq', 'function_label', 'image_ids', 'image_manu', 'reflux_label', 'surgery_label', 'view_label']
	for csv_path in csv_list:
		data = pd.read_csv(csv_path)
		data.columns = header_names

		if len(data) == 0:
			continue

		# Split the image_ids column into subj_id, scan_num, image_num columns
		data[['subj_id','scan_num', 'image_num']] = data.image_ids.str.split('_', expand=True) 


		all_labels = all_labels.append(data, ignore_index=True)
	return all_labels


def analyze_labels(df, output_base):
	# How many scans do we have person?
	scans_per_person = df.groupby('subj_id').scan_num.agg(['nunique'])
	scans_per_person.hist()
	plt.ylabel("Number of Patients")
	plt.xlabel("Number of Scans")
	plt.title("Number of Scans per Patient")
	plt.savefig(os.path.join(output_base, 'scans_per_person.jpg'))

	# How many images have view_labels?
	view_labels = df['view_label'].value_counts()
	view_labels.to_csv(os.path.join(output_base, 'view_labels.csv'))

	# Of the images that have view labels, how many also have target labels?
	pass


def run_analysis(ROOT_DIR):
	label_dir = os.path.join(ROOT_DIR, "all_label_csv")
	image_dir = os.path.join(ROOT_DIR, "all_label_img")
	output_base_dir = os.path.join(ROOT_DIR, "output")
	ALL_LABEL_DF_CSV = os.path.join(ROOT_DIR, "all_label_df.csv")

	os.makedirs(output_base_dir, exist_ok=True)


	all_label_csvs = glob.glob(os.path.join(label_dir, "1*.csv"))

	if not os.path.exists(ALL_LABEL_DF_CSV):
		print("Extracting CSVs to DF")
		label_df = extract_label_csvs_to_df(all_label_csvs)
		label_df.to_csv(ALL_LABEL_DF_CSV)

	label_df = pd.read_csv(ALL_LABEL_DF_CSV)

	analyze_labels(label_df, output_base_dir)

if __name__ == "__main__":
	run_analysis(ROOT_DIR)
