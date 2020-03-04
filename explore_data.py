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
	# How many scans do we have per person?
	scans_per_person = df.groupby('subj_id').scan_num.agg(['nunique'])
	scans_per_person.hist()
	plt.ylabel("Number of Patients")
	plt.xlabel("Number of Scans")
	plt.title("Number of Scans per Patient")
	plt.savefig(os.path.join(output_base, 'scans_per_person.jpg'))

	# How many images have view_labels?
	view_labels = df['view_label'].value_counts()
	view_labels.to_csv(os.path.join(output_base, 'view_labels.csv'))

	# Of the images that have a target label, how many have a view label?
	least_one_target = df[(df.function_label != "Missing") | (df.reflux_label != "Missing") | (df.surgery_label != "Missing")]
	at_least_one_target_missing_view = least_one_target[(least_one_target.view_label == "Missing") | (least_one_target.view_label == "Other")]
	scan_counts = at_least_one_target_missing_view.groupby('subj_id').scan_num.agg(['nunique']).sum()
	least_one_target_scan_counts = least_one_target.groupby('subj_id').scan_num.agg(['nunique'])

	print("="*100)
	print(f"There are {len(least_one_target)} images from {least_one_target_scan_counts['nunique'].sum()} scans from {least_one_target.subj_id.nunique()} patients that have at least one target label")
	print(f"Of the {len(least_one_target)} images that have at least one target label, {len(at_least_one_target_missing_view)} are missing a view label.")
	print(f"Therefore, we are missing view labels for {scan_counts['nunique']} scans from {at_least_one_target_missing_view.subj_id.nunique()} patients (which have target labels).")
	print(f"Of the patients that have at least one target label, {len(least_one_target_scan_counts[least_one_target_scan_counts['nunique'] > 1])} have more than one scan (potential improvement in lstm)")


	# Of the images that have view labels, how many also have target labels?
	# Exclude  "Missing" and "Other"
	labelled_view_df = df[(df.view_label != "Missing") & (df.view_label != "Other")]

	# How images have at least one label?
	labelled_with_at_least_one_target = labelled_view_df[(labelled_view_df.function_label != "Missing") | (labelled_view_df.reflux_label != "Missing") | (labelled_view_df.surgery_label != "Missing")]
	scan_counts = labelled_with_at_least_one_target.groupby('subj_id').scan_num.agg(['nunique']).sum()
	print("="*100)
	print(f"Of the {len(labelled_view_df)} images that have a view label, {len(labelled_with_at_least_one_target)} have at least one target label.")
	print(f"Therefore, we have images with a view label and at least one target label for {scan_counts['nunique']} scans from {labelled_with_at_least_one_target.subj_id.nunique()} patients.")


	print("="*100)
	labels_list = ['function_label', 'reflux_label', 'surgery_label']
	for ind in range(len(labels_list)):
		labelled_view_with_target_df = labelled_view_df[labelled_view_df[labels_list[ind]] != "Missing"]
		scan_counts = labelled_view_with_target_df.groupby('subj_id').scan_num.agg(['nunique']).sum()
		print(f"Of the {len(labelled_view_df)} images that have a view label, {len(labelled_view_with_target_df)} images from {scan_counts['nunique']} scans (from {labelled_view_with_target_df.subj_id.nunique()} patients) have a {labels_list[ind]} target label.")


	# Check that surgery and reflux come in pairs, and that the images with a function label are a subset of these
	view_labelled_with_ref_and_surg = labelled_view_df[(labelled_view_df.reflux_label != "Missing") & (labelled_view_df.surgery_label != "Missing")]
	view_labelled_with_all_three_labels = labelled_view_df[(labelled_view_df.function_label != "Missing") & (labelled_view_df.reflux_label != "Missing") & (labelled_view_df.surgery_label != "Missing")]
	print(f"Of the {len(view_labelled_with_ref_and_surg)} images that have both reflux and surgery labels, {len(view_labelled_with_all_three_labels)} also have a function label.")


def run_analysis(root_dir):
	label_dir = os.path.join(root_dir, "all_label_csv")
	image_dir = os.path.join(root_dir, "all_label_img")
	output_base_dir = os.path.join(root_dir, "output")
	ALL_LABEL_DF_CSV = os.path.join(root_dir, "all_label_df.csv")

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
