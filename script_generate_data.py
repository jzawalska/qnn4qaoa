from data_generator import generate_data

for i in range(1, 11):

    samples_range = (0, 20)
    type = 'test' # 'training'
    generate_data(i, samples_range, type, '4')