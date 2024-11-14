import os
import sys
import time
import argparse
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, NoSuchElementException, ElementClickInterceptedException
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import torch


def configure_driver():
    options = webdriver.ChromeOptions()
    # Uncomment options below if needed
    # options.add_argument('--window-size=800,600')
    # options.add_argument('--window-position=2000,0')
    # options.add_argument('--no-sandbox')
    # options.add_argument('--disable-gpu')
    # options.add_argument('--disable-infobars')
    # options.add_argument('--disable-extensions')
    return webdriver.Chrome(options=options)


def login(driver, job_portal_url, username, password):
    print("start login")
    if job_portal_url is not None:
        driver.get(job_portal_url)
        try:
            # Log in process with retries
            for attempt in range(3):
                try:
                    WebDriverWait(driver, 30).until(
                        EC.element_to_be_clickable((By.ID, 'login_Layer'))
                    ).click()
                    break
                except (TimeoutException, ElementClickInterceptedException):
                    print(f"Attempt {attempt + 1}: Login button click failed, retrying...")
                    driver.refresh()
                    continue

            # Wait for email input and enter username
            email_input = WebDriverWait(driver, 30).until(
                EC.visibility_of_element_located(
                    (By.XPATH, "//label[contains(text(), 'Email ID / Username')]/following-sibling::input"))
            )
            email_input.clear()
            email_input.send_keys(username)

            # Wait for password input and enter password
            password_input = WebDriverWait(driver, 30).until(
                EC.visibility_of_element_located(
                    (By.XPATH, "//label[contains(text(), 'Password')]/following-sibling::input"))
            )
            password_input.clear()
            password_input.send_keys(password)

            # Click login button
            login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]")
            WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Login')]")))
            login_button.click()

            # Wait for the login to complete
            #time.sleep(2)  # Give some time for login to fully complete
            print("Login successful")
            time.sleep(5)
        except TimeoutException as e:
            print("Login timeout error:", e)
            sys.exit(1)
        except NoSuchElementException as e:
            print("Element not found error during login:", e)
            sys.exit(1)
        except Exception as e:
            print("Login error:", e)
            sys.exit(1)


def search_jobs(driver, jobs_urls):
    print("start search")
    all_job_links = []
    for counter, jobs_url in enumerate(jobs_urls, start=1):
        try:
            # Open each pre-configured jobs URL
            if jobs_url is not None:
                driver.get(jobs_url)
                print(f"Jobs page loaded successfully for URL: {jobs_url}")
                print(f"Page {counter} loaded")
                time.sleep(1)
                job_links = collection(driver)
                all_job_links.extend(job_links)
                time.sleep(2)
        except Exception as e:
            print(f"Error while navigating to jobs page {jobs_url}:", e)
    print("All collected job links:", all_job_links)
    return all_job_links


def collection(driver):
    print("Collecting data from the page...")
    job_links = []
    try:
        # Wait for job elements to be present
        WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'srp-jobtuple-wrapper'))
        )
        # Find all job elements by class name 'srp-jobtuple-wrapper'
        job_elements = driver.find_elements(By.CLASS_NAME, 'srp-jobtuple-wrapper')
        print("element start")
        for counter, job_element in enumerate(job_elements, start=0):
            print(job_element)
            print(f"element counter:{counter} ")
            job_id = job_element.get_attribute('data-job-id')
            job_details = job_element.find_element(By.CLASS_NAME, 'cust-job-tuple.layout-wrapper.lay-2.sjw__tuple ')
            # Extract data from row1
            row1 = job_details.find_element(By.CLASS_NAME, 'row1')
            job_title_element = row1.find_element(By.XPATH, ".//a[@class='title ']")
            job_href = job_title_element.get_attribute('href')
            job_position = job_title_element.text
            # Extract data from row2
            row2 = job_details.find_element(By.CLASS_NAME, 'row2')
            company_details = row2.find_element(By.CLASS_NAME, 'comp-dtls-wrap')
            company_name_element = company_details.find_element(By.CLASS_NAME, 'comp-name')
            company_name = company_name_element.text
            company_href = company_name_element.get_attribute('href')
            job_links.append((company_name, job_position, job_href, company_href))
            print(f"Company: {company_name}, Job Position: {job_position}, Job Link: {job_href}, Company Link: {company_href}")
        print("element END")
    except NoSuchElementException as e:
        print("Error while collecting job data:", e)
    return job_links


def write_to_csv(job_links):
    date_str = datetime.now().strftime('%Y-%m-%d')
    csv_filename = f"NAUKRI_job_links_{date_str}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Company Name', 'Job Position', 'Job URL', 'Company URL'])
        writer.writerows(job_links)
    print(f"Job links written to {csv_filename}")
    return csv_filename


def write_company_jobs_to_csv(company_job_links):
    date_str = datetime.now().strftime('%Y-%m-%d')
    csv_filename = f"NAUKRI_COMPANY_job_links_{date_str}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Company Name', 'Job Position', 'Company Job URL'])
        for company_name, job_position, job_link in company_job_links:
            writer.writerow([company_name, job_position, job_link])
    print(f"Company job links written to {csv_filename}")


def navigate_to_company_sites(driver, csv_filename):
    print("Navigating to company sites...")
    all_company_job_links = []
    try:
        with open(csv_filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                company_url = row['Company URL']
                print(f"Navigating to Company URL: {company_url}")
                if company_url is not None:
                    driver.get(company_url)
                    time.sleep(2)  # Pause to allow the page to load properly
                    company_job_links = collect_company_jobs(driver)
                    all_company_job_links.extend(company_job_links)
        write_company_jobs_to_csv(all_company_job_links)
    except FileNotFoundError as e:
        print(f"CSV file not found: {csv_filename}")
    except Exception as e:
        print(f"Error while navigating to company sites: {e}")


def collect_company_jobs(driver):
    print("Collecting job data from the company page...")
    company_job_links = []
    try:
        # Wait for job elements to be present
        WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'srp-jobtuple-wrapper'))  # Updated class name for job listings
        )
        # Find all job elements by class name 'srp-jobtuple-wrapper'
        job_elements = driver.find_elements(By.CLASS_NAME, 'srp-jobtuple-wrapper')
        print("Company jobs element start")
        for counter, job_element in enumerate(job_elements, start=0):
            print(job_element)
            print(f"element counter:{counter} ")
            job_id = job_element.get_attribute('data-job-id')
            # Extract data from job listing
            job_title_element = job_element.find_element(By.XPATH, ".//a[contains(@class, 'title ')]")
            job_href = job_title_element.get_attribute('href')
            job_position = job_title_element.text
            company_details = job_element.find_element(By.CLASS_NAME, 'comp-dtls-wrap')
            company_name_element = company_details.find_element(By.CLASS_NAME, 'comp-name')
            company_name = company_name_element.text
            company_job_links.append((company_name, job_position, job_href))
            print(f"Company: {company_name}, Job Position: {job_position}, Job Link: {job_href}")
        print("Company jobs element END")
    except NoSuchElementException as e:
        print("Error while collecting company job data:", e)
    return company_job_links


def filter_lead_positions():
    date_str = datetime.now().strftime('%Y-%m-%d')
    input_filename = f"NAUKRI_COMPANY_job_links_{date_str}.csv"
    output_filename = f"NAUKRI_LEAD_job_links_{date_str}.csv"
    try:
        with open(input_filename, mode='r', encoding='ISO-8859-1') as infile, open(output_filename, mode='w', newline='', encoding='ISO-8859-1') as outfile:
            reader = csv.DictReader(infile)
            writer = csv.writer(outfile)
            writer.writerow(['Company Name', 'Job Position', 'Company Job URL'])
            for row in reader:
                if 'lead' in row['Job Position'].lower():
                    writer.writerow([row['Company Name'], row['Job Position'], row['Company Job URL']])
        print(f"Filtered lead positions written to {output_filename}")
    except FileNotFoundError as e:
        print(f"CSV file not found: {input_filename}")
    except Exception as e:
        print(f"Error while filtering lead positions: {e}")

def rank_lead_positions():
    date_str = datetime.now().strftime('%Y-%m-%d')
    date_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    input_filename = f"NAUKRI_LEAD_job_links_{date_str}.csv"
    output_filename = f"NAUKRI_RANKED_LEAD_job_links_{date_str}.csv"
    try:
        with open(input_filename, mode='r', encoding='ISO-8859-1') as infile, open(output_filename, mode='w', newline='',encoding='ISO-8859-1') as outfile:
            reader = csv.DictReader(infile)
            # Validate if expected headers are present
            expected_headers = {'Company Name', 'Job Position', 'Company Job URL'}
            if not expected_headers.issubset(reader.fieldnames):
                raise ValueError(f"Input CSV is missing one or more expected columns: {expected_headers}")

            writer = csv.writer(outfile)
            writer.writerow(['Rank', 'Company Name', 'Job Position', 'Job URL', 'GenAI Score', 'Data Score'])
            ranked_jobs = []


            for row in reader:
                try:
                    job_position = row['Job Position']
                    print(f"job position {job_position}")
                    # Calculate similarity scores
                    similarity_scores_list = []
                    for keyword in ['GenAI', 'Data']:
                        scores = calculate_similarity_scores(keyword, job_position)
                        if "error" in scores:
                            raise ValueError(
                                f"Error calculating similarity for keyword '{keyword}' in job '{job_position}': {scores['error']}")
                        similarity_scores_list.append(scores['relevance_score'])

                    # Assign the calculated scores to the row
                    #row['Lead Score'] = similarity_scores_list[0]
                    #row['GenAI Score'] = similarity_scores_list[1]
                    #row['AI Score'] = similarity_scores_list[2]
                    #row['Data Score'] = similarity_scores_list[3]
                    row['GenAI Score'] = similarity_scores_list[0]
                    row['Data Score'] = similarity_scores_list[1]

                    ranked_jobs.append(row)
                except KeyError as e:
                    print(f"Missing expected column in row: {e}")
                except ValueError as e:
                    print(e)
                except Exception as e:
                    print(f"Unexpected error processing row: {e}")

            # Sort the jobs based on the average of scores or a chosen score
            ranked_jobs = sorted(ranked_jobs, key=lambda x: (
                                                                     x['GenAI Score'] + x['Data Score']
                                                            ) / 2, reverse=True)

            # Write sorted jobs to the output file with rank
            for rank, row in enumerate(ranked_jobs, start=1):
                if row['GenAI Score'] > 50 or row['Data Score'] > 50 :
                    writer.writerow([
                        rank, row['Company Name'], row['Job Position'], row['Company Job URL'],
                        row['GenAI Score'], row['Data Score']
                    ])

        print(f"Ranked lead positions written to {output_filename}")

    except FileNotFoundError as e:
        print(f"CSV file not found: {input_filename}")
    except ValueError as e:
        print(f"Input file error: {e}")
    except Exception as e:
        print(f"Error while ranking lead positions: {e}")


def calculate_similarity_scores(resume_text, job_description):
    try:
        # Load the model for embedding-based similarity (e.g., 'paraphrase-mpnet-base-v2')
        model = SentenceTransformer('paraphrase-mpnet-base-v2')

        # Encode both texts for cosine similarity and relevance score
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        job_embedding = model.encode(job_description, convert_to_tensor=True)

        # 1. Cosine Similarity
        cosine_similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()

        # 2. Relevance Score (converted to percentage from cosine similarity)
        relevance_score = cosine_similarity_score * 100

        # 3. Jaccard Similarity for Keywords
        resume_keywords = set(resume_text.lower().split())  # Simplified keyword extraction for demonstration
        job_keywords = set(job_description.lower().split())
        jaccard_similarity_score = len(resume_keywords.intersection(job_keywords)) / len(
            resume_keywords.union(job_keywords))

        # 4. TF-IDF Weighted Cosine Similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        tfidf_cosine_similarity_score = sklearn_cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]).item()

        # 5. Additional Similarity Score
        # Here, we use the same cosine similarity as an example of an additional similarity metric,
        # but this can be replaced with another similarity metric as desired.
        additional_similarity_score = cosine_similarity_score  # Reusing cosine similarity for demonstration

        # Combine all scores with specified weights
        combined_fit_score = (
                0.4 * cosine_similarity_score +
                0.2 * (relevance_score / 100) +  # Convert relevance back to scale of 0 to 1
                0.15 * jaccard_similarity_score +
                0.15 * tfidf_cosine_similarity_score +
                0.1 * additional_similarity_score
        )

        # Return results
        return {
            "cosine_similarity_score": cosine_similarity_score,
            "relevance_score": relevance_score,
            "jaccard_similarity_score": jaccard_similarity_score,
            "tfidf_cosine_similarity_score": tfidf_cosine_similarity_score,
            "additional_similarity_score": additional_similarity_score,
            "combined_fit_score": combined_fit_score
        }

    except Exception as e:
        return {"error": str(e)}

def find_apply_type(driver):
    print("writing apply types for normal job list")
    date_str = datetime.now().strftime('%Y-%m-%d')
    first_file = f"NAUKRI_job_links_{date_str}.csv"
    first_out = f"NAUKRI_APPLY_0_job_links_{date_str}.csv"
    try:
        with open(first_file, mode='r', encoding='ISO-8859-1') as infile, open(first_out, mode='w', newline='',encoding='ISO-8859-1') as outfile:
            reader = csv.DictReader(infile)
            writer = csv.writer(outfile)
            writer.writerow(['Company Name', 'Job Position', 'Apply Type', 'Job URL'])


            for row in reader:
                job_link = row['Job URL']
                print(job_link)
                driver.get(job_link)
                time.sleep(1)
                try:
                    # Check for apply button container
                    apply_containers = driver.find_elements(By.CLASS_NAME, 'styles_jhc__apply-button-container__5Bqnb')
                    if not apply_containers:
                        print("No apply container found.")
                        continue

                    applytype = ''

                    for apply_container in apply_containers:
                        # Check for different apply types
                        if apply_container.find_elements(By.CLASS_NAME, "styles_apply-button__uJI3A"):
                            print("Apply type: SIMPLE APPLY")
                            applytype = 'SIMPLE APPLY'
                        elif apply_container.find_elements(By.CLASS_NAME, "styles_company-site-button__C_2YK"):
                            print("Apply type: COMPANY APPLY")
                            applytype = 'COMPANY APPLY'
                        elif apply_container.find_elements(By.CLASS_NAME, "styles_already-applied__4KDhw"):
                            print("Apply type: ALREADY APPLIED")
                            applytype = 'ALREADY APPLIED'
                        else:
                            print("No recognized apply type found.")

                        writer.writerow([row['Company Name'], row['Job Position'], applytype, row['Job URL']])

                except NoSuchElementException:
                    print("Apply button container not found.")

    except FileNotFoundError:
        print(f"File not found: {first_file}")
    except Exception as e:
        print(f"Exception while adding apply type: {e}")


# def apply(driver, url):
#     try:
#         # Attempt to click the apply button within the container
#         driver.get(url)
#         time.sleep(1)
#         try:
#             apply_button = driver.find_element(By.CLASS_NAME, "styles_apply-button__uJI3A")
#             apply_button.click()
#             time.sleep(1)  # Allow time for potential popups or redirects
#
#             # Check if the chatbot window opened
#             if driver.find_elements(By.CLASS_NAME, "chatbot_DrawerContentWrapper"):
#                 print("Chat window opened after applying.")
#                 return 2
#
#             print("Successfully applied to the job using SIMPLE APPLY.")
#             time.sleep(2)  # Wait for any potential redirects or follow-up actions
#             return 1
#         except NoSuchElementException:
#             # Check if the job is already applied to
#             if driver.find_elements(By.CLASS_NAME, "styles_already-applied__4KDhw"):
#                 print("Job already applied to.")
#                 return 3
#             else:
#                 print("Apply button not found within the container.")
#     except Exception as e:
#         print(f"Error while attempting to apply: {e}")
#     return 0

def apply(driver, url):
    try:
        # Attempt to click the apply button within the container
        if url is not None:
            driver.get(url)
            time.sleep(1)
            try:
                apply_button = driver.find_element(By.CLASS_NAME, "styles_apply-button__uJI3A")
                apply_button.click()
                time.sleep(1)  # Allow time for potential popups or redirects

                # Check if the chatbot window opened
                if driver.find_elements(By.CLASS_NAME, "chatbot_DrawerContentWrapper"):

                    #REMOVE ME
                    #return 2

                    print("Chat window opened after applying.")
                    # Wait for the user to manually enter details in the chat window
                    while driver.find_elements(By.CLASS_NAME, "chatbot_DrawerContentWrapper"):
                        print("Chat window is still open. Please enter the required details.")
                        time.sleep(5)  # Check every 5 seconds if the chat window is still open
                    print("Chat window closed, assuming details have been entered.")
                    return 2

                print("Successfully applied to the job using SIMPLE APPLY.")
                time.sleep(2)  # Wait for any potential redirects or follow-up actions
                return 1
            except NoSuchElementException:
                # Check if the job is already applied to
                if driver.find_elements(By.CLASS_NAME, "styles_already-applied__4KDhw"):
                    print("Job already applied to.")
                    return 3
                else:
                    print("Apply button not found within the container.")
    except Exception as e:
        print(f"Error while attempting to apply: {e}")
    return 0



def filter_simple_apply_jobs(driver):
    date_str = datetime.now().strftime('%Y-%m-%d')
    date_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    input_file = f"NAUKRI_APPLY_0_job_links_{date_str}.csv"
    output_file = f"NAUKRI_APPLY_1_job_links_{date_time_str}.csv"
    try:
        with open(input_file, mode='r', encoding='ISO-8859-1') as infile, open(output_file, mode='w', newline='', encoding='ISO-8859-1') as outfile:
            reader = csv.DictReader(infile)
            writer = csv.writer(outfile)
            writer.writerow(['Company Name', 'Job Position', 'Apply Type', 'Status', 'Job URL'])

            for row in reader:
                if row['Apply Type'] == 'SIMPLE APPLY':
                    print(row['Job URL'])
                    apply_status = apply(driver,str(row['Job URL']))
                    if apply_status == 1:
                        writer.writerow([row['Company Name'], row['Job Position'],"APPLIED", row['Apply Type'], row['Job URL']])
                    elif apply_status == 2:
                        writer.writerow(
                            [row['Company Name'], row['Job Position'], "CHAT WINDOW", row['Apply Type'], row['Job URL']])
                    else:
                        writer.writerow(
                            [row['Company Name'], row['Job Position'], "FAILED", row['Apply Type'], row['Job URL']])

        print(f"Filtered SIMPLE APPLY jobs written to {output_file}")
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"Error while filtering SIMPLE APPLY jobs: {e}")






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True, help='Username for login')
    parser.add_argument('--word', required=True, help='Password for login')
    args = parser.parse_args()

    job_portal_url = 'https://www.naukri.com'
    # jobs_urls = [
    #     "https://www.naukri.com/senior-data-engineer-jobs?k=senior+data+engineer&nignbevent_src=jobsearchDeskGNB&wfhType=2",
    #     "https://www.naukri.com/senior-data-engineer-jobs-2?k=senior+data+engineer&nignbevent_src=jobsearchDeskGNB&wfhType=2",
    #     "https://www.naukri.com/senior-data-engineer-jobs-3?k=senior+data+engineer&nignbevent_src=jobsearchDeskGNB&wfhType=2",
    #     "https://www.naukri.com/senior-data-engineer-jobs-4?k=senior+data+engineer&nignbevent_src=jobsearchDeskGNB&wfhType=2",
    #     "https://www.naukri.com/senior-data-engineer-jobs-5?k=senior+data+engineer&nignbevent_src=jobsearchDeskGNB&wfhType=2"
    #
    # ]
    jobs_urls = [
        "https://www.naukri.com/senior-data-engineer-jobs?k=senior+data+engineer&nignbevent_src=jobsearchDeskGNB&wfhType=2"
    ]

    driver = configure_driver()
    try:
        login(driver, job_portal_url, args.user, args.word)
        all_job_links = search_jobs(driver, jobs_urls)
        print("Final collected job links:", all_job_links)
        csv_filename = write_to_csv(all_job_links)
        navigate_to_company_sites(driver, csv_filename)
        filter_lead_positions()
        rank_lead_positions()
        #login(driver, job_portal_url, args.user, args.word)
        find_apply_type(driver)
        link = "https://www.naukri.com/job-listings-senior-data-engineer-pune-rudder-analytics-rudder-analytics-pune-3-to-6-years-311024003925"
        filter_simple_apply_jobs(driver)




    finally:
        driver.quit()


if __name__ == "__main__":
    main()
    #runtime params
    #--user username --word password
