import os.path
from vmtk import pypes

def Frangi_filter(file, sigma_min=2.0, sigma_max=6.0, sigma_steps=None, alpha=0.5,
				  beta=500, gamma=500, prefix='Frangi_', info_file=None, d_results=None,
				  no_output=False):

	# first create pype string

	if sigma_steps is None:
		sigma_steps = (sigma_max - sigma_min)

	input_file = os.path.abspath(file).replace('\\', '/')

	file = os.path.basename(file)

	if d_results is not None:
		if not os.path.exists(d_results):
			os.makedirs(d_results)
		output_dir = os.path.abspath(d_results).replace('\\', '/')
	else:
		output_dir = os.path.dirname(input_file)

	ofile = output_dir + '/' + prefix + file + 'f'

	pype_part_1 = ['vmtkimagereader -f tiff -ifile ']
	pype_part_2 = [(' --pipe vmtkimagecast -type float'
					' --pipe vmtkimagevesselenhancement -sigmamin {0} '
					'-sigmamax {1} -sigmasteps {2} -alpha {3} -beta {4} -gamma {5} '
					'--pipe vmtkimageshiftscale -type ushort -mapranges 1 '
					'-inputrange 0.0 1.0 -outputrange 0.0 65535.0 -ofile ').format(
		sigma_min, sigma_max,
		sigma_steps, alpha,
		beta, gamma)]

	if info_file is not None:
		pype_arg = pype_part_1[0] + input_file + pype_part_2[0] + ofile
		f = open(info_file, "a")
		f.write(pype_arg)
		f.close()

	# execute command
	pype_argument = pype_part_1[0] + input_file + pype_part_2[0] + ofile
	print(pype_argument)

	if not no_output:
		pypes.PypeRun(pype_argument)