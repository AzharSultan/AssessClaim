import logging
import argparse
import pandas as pd
from ragatouille import RAGPretrainedModel
from utils import get_record_df, get_initial_answers, get_verification_statements, extract_n_digit_number
from utils import extract_cpt, extract_age, extract_cons_treatment, extract_polyposis
from utils import extract_symptomatic, extract_prior_colonoscopy, extract_cancer_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def review_case(answers, statements, rag_model, medical_history, match_threshold=16, cpt=45378, record_year=2023):
    treat_statements = [statements[3], statements[6], statements[8]]
    age, age_conf, age_evid = extract_age(answers[0], statements[0], rag_model, match_threshold, record_year)
    is_cpt, cpt_conf, cpt_evid = extract_cpt(answers[1], statements[1], rag_model, cpt=cpt, match_threshold=match_threshold)
    is_continue, treat_reason, treat_conf, treat_evid = extract_cons_treatment(answers[7], treat_statements, medical_history, rag_model, match_threshold)
    is_polyp, polyp_conf, polyp_evid = extract_polyposis(answers[9], statements[7], rag_model, match_threshold)
    is_symp, symp_conf, symp_evid = extract_symptomatic(answers[2:6], statements[4],rag_model, match_threshold)
    is_colon, colon_conf, colon_evid = extract_prior_colonoscopy(answers[6], statements[2], rag_model, match_threshold)
    is_cancer, cancer_conf, cancer_evid = extract_cancer_history(answers[8], statements[5], rag_model, match_threshold)

    outcome_conf = min([cpt_conf, age_conf])

    df = pd.DataFrame.from_records([('age', age, age_evid, age_conf),
                                    ('cpt', is_cpt, cpt_evid, cpt_conf),
                                    ('prior treatment', is_continue, treat_reason, treat_conf),
                                    ('polyposis', is_polyp, polyp_evid, polyp_conf),
                                    ('stmptomatic', is_symp, symp_evid, symp_conf),
                                    ('prior colonoscopy', is_colon, colon_evid, colon_conf),
                                    ('family cancer history', is_cancer, cancer_evid, cancer_conf),
                                    ('OUTCOME', 'Declined', '', outcome_conf)],
                                    columns=['Criteria', 'Value', 'Reason', 'Confidence'])

    df = df.set_index('Criteria', drop=True)

    if not is_cpt:
        df.loc['OUTCOME', 'Reason'] = f'{cpt} is not identified among the requested procedures'
        df.loc['OUTCOME', 'Value'] = 'Declined'
        return df

    df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], treat_conf])
    if not is_continue:
        df.loc['OUTCOME', 'Reason'] = treat_reason
        return df

    if is_polyp:
        df.loc['OUTCOME', 'Reason'] = f'Patient of age {age} with family history of colonic adenomatous polyposis'
        df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], polyp_conf])
        df.loc['OUTCOME', 'Value'] = 'Approved'
        return df
    
    if age <= 21:
        if is_symp:
            df.loc['OUTCOME', 'Reason'] = f'Pediatric patient of age {age} with symptoms as mentioned in report here:\n {symp_evid}'
            df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], symp_conf])
            df.loc['OUTCOME', 'Value'] = 'Approved'
            return df
        else:
            df.loc['OUTCOME', 'Reason'] = f'Pediatric patient of age {age} with no relevant §§1symptoms does not match criteria for the procedure 45378'
            df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], polyp_conf, symp_conf])
            return df
        
    elif age >= 40:
        if not is_colon:
            df.loc['OUTCOME', 'Reason'] = f'Patient of age {age} with no colonoscopy in last 10 years'
            df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], colon_conf])
            df.loc['OUTCOME', 'Value'] = 'Approved'
            return df
        
        if age >= 45:
            if is_cancer:
                if is_symp:
                    # find symptoms
                    df.loc['OUTCOME', 'Reason'] = f'Patient of age {age} with symptoms and family history of colorectal cancer as mentions in report here:\n'\
                                                  f'{cancer_evid}\n'\
                                                  f'{symp_evid}'
                    df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], cancer_conf, symp_conf])
                    df.loc['OUTCOME', 'Value'] = 'Approved'
                    return df
                else:
                    df.loc['OUTCOME', 'Reason'] = f'Patient of age {age} with family history of colorectal cancer, colonoscopy in last 10 years but no relevant symptoms'
                    df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], cancer_conf, symp_conf, colon_conf, polyp_conf])
                    return df
            else:
                df.loc['OUTCOME', 'Reason'] = f'Patient of age {age} with no family history of colorectal cancer and had colonoscopy in last 10 years as mentioned in report here:\n'\
                                              f'{colon_evid}'
                df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], cancer_conf, colon_conf, polyp_conf])
                return df
        else:
            df.loc['OUTCOME', 'Reason'] = f'Patient of age {age} had colonoscopy in last 10 years as mentioned in report here:\n'\
                                          f'{colon_evid}'
            df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], colon_conf, polyp_conf])
            return df
    else:
        df.loc['OUTCOME', 'Reason'] = f'Patient of age {age} with no family history of colonic adenomatous polyposis of unknown etiology'
        df.loc['OUTCOME', 'Confidence'] = min([df.loc['OUTCOME', 'Confidence'], polyp_conf])
        return df

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('record_file')
    parser.add_argument('output_file')
    parser.add_argument('-c', '--cpt', type=int, default=45378)
    parser.add_argument('-m', '--match_threshold', type=float, default=16.0)
    parser.add_argument('-y', '--record_year', type=int, default=2023)

    args = parser.parse_args()

    record = get_record_df(args.record_file)
    medical_history = '\n'.join(record.text)
    initial_answers = get_initial_answers(medical_history)
    birth_year = extract_n_digit_number(initial_answers[0], 4)
    statements = get_verification_statements(medical_history, birth_year)

    filtered_record = record[record.text.apply(lambda x: len(x.replace(' ', '')) > 10) &
                    record.type.apply(lambda x: 'Heading' not in x.name)]
    rag_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    index_path = rag_model.index(index_name="helm", collection=list(filtered_record.text))

    df = review_case(initial_answers, statements, rag_model, medical_history, 
                     args.match_threshold, args.cpt, args.record_year)
    
    logger.info(f'{df}')
    df.to_csv(args.output_file)


