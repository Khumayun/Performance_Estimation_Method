import analysis
import sys

def usage():
	print("""
        This file consist of several functions(to run all functions and get results use -all).
        Use -d path_to_dataset.csv to explicitly define input dataset location.
        It performs full analysis used in the paper and export all results to results folder.""")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

	filename = 'dataset/tom_project_metrics_27.03.2022.csv'
	try:
		if len(sys.argv) == 1:
			analysis.full_analysis(filename)
		elif sys.argv[1] == '-d':
			filename = sys.argv[2]

	except FileNotFoundError:
		print('File does not exist')
	except IndexError:
		usage()
