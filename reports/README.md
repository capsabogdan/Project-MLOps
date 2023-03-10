---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [x] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 17 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s212558, s210172, s184419, s213158, s213161 ---

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We used the third-party framework Pytorch-Geometric, which is a deep learning framework for geometric data such as graphs. This framework was extremely beneficial for us as we were working with graphical data - movie predictions based on user ratings. We used Pytorch-Geometric to set up the graphs when we were making the datasets. Furthermore, from Pytorch-Geometric we used the SAGEConv class in the GNNEncoder part of our model. In our projet we also use the Pytorch framework, which made the handling of tensors convenient. We also used the torch.nn.Linear class for making the FC layers in our EdgeDecoder part of our model. ---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

--- We used the pipreqs package, as well as some manual additions, to manage our dependencies. We ran the pipreqs command in our project to generate a 'requirements.txt' file with the libraries, and their versions, required for our project. We experienced some problems with some of the libraries such as torch-geometric, so for those we would sometimes have to manually change the order and version of the install. We also have a 'requirements_docker_train.txt' file, which is used when making a docker image. To get a complete copy of our dependencies, one would have to make a new empty python environment, go to the project root, and run "pip install -r requirements.txt" ---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- We used the cookiecutter template to initialize our project. In the src/models folder we have made a 'model.py' file which describes our model, a 'train_model.py' script for training the model, and a 'predict_model.py' script for predictng. In the '_init_.py' file we have included some paths. We have filled out the 'make_dataset.py' script in the src/data folder such that it takes the raw data and processes it, and then placed the processed data in the data/processed folder. The data processing includes some data enriching, which is done with the help of the 'oasis.py' script. We use both the raw and processed data folders, but we have removed the external and interim folders. ---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- We wrote our code following the pep8 protocol, to have a single style for our whole project and make it easier for all our members to read the other members' code. We also used isort for sorting our imports. For the 'make_dataset.py' script, we implemented code from this [this](https://medium.com/arangodb/integrate-arangodb-with-pytorch-geometric-to-build-recommendation-systems-dd69db688465) blog post, thus we decided to just leave the code format as it was. Following a specific code format such as pep8 is important in large projects because this means that code written from different users will be in the same format and thus more understandable. ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- In total we have implemented 10 tests in our code, which are all related to the data of our project. The first 4 tests are checking that all labels are present in our dataset. The remaining 6 tests are related to the size of the data, with some making sure there are the right number of movies and users, and some checking that the number of edges are also correct. ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- Our total code coverage is 100%, which suggests that our tests cover our whole code. Since we only have tests related to the dataset, this seems to be too high, and should not be used to imply that our code is error free. Our tests are related to the dataset, and are thus not taking into consideration the model architecture for example which could be a problem. Therefore, a total code coverage of 100% does not necessarily mean that the code is error free, it just means that the tests cover it all. But the better the tests are, the more likely the code is to be error free. ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- In our GitHub repository we used three branches: master, test, and dev. In our master branch we have our producting-ready code, which is code that we know is running as it should (but at the start of the project it didn't have all the right functionality). When we were developing our model, data, training, etc. we would all push to the dev branch. When pushing to the dev branch no tests are performed, making the process faster. This is useful when we are just making small changes or fixes, e.g. spelling mistakes. We would then merge the dev branch to the test branch, where we would perform our tests. This process takes a bit of time, but since we don't do it as often this is not a problem. When we have code that we are satisfied with in our test branch, we would send a pull request to the master branch. ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- We used DVC for pulling the raw data to our local machines from a Cloud Bucket, and then used it for pushing the processed data to a separate Bucket. Even though our dataset wasn???t particularly large it was convenient for us to have version control, since we were able to store it in a remote location, and simply use dvc pull to get the data to our local machine. Another benefit is that if we were to change our dataset (i.e.add more data to it), we could still store the previous version of the dataset in case we wanted to replicate an earlier experiment.---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- For the CI we have set up multiple branches and Github actions. The CI is done for testing and the actions get triggered when merging from **DEV** to **TEST**.

We are mainly testing the data, asserting that all labels are represented and the size of each datapoint. Furthermore, we are running the CI on the latest Ubuntu version, python 3.8. So code changes can pass when pushing to dev, but for the merge to test should be evaluated and be aproved only if they do not affect the size of the datapoints and labels in use. The final step, which leads to production is to do a PR from test to main, as this will trigger further the CD in the Cloud. 

Our last Github Action: https://github.com/capsabogdan/Project-MLOps/actions/runs/3961993677 ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- We used config files in our project to configure experiments. We made different config files named ???exp1???, ???exp2???, ... where each file contains information such as batch_size, num_epochs, etc. We then also have a ???default_config??? file which chooses which experiment to run. For a user to run a new experiment they would have to make a new ???expX??? file with the desired configurations, and then change the ???default_config??? file to choose that experiment, and then they just run the ???train_model.py??? script. ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- We have a folder called ???outputs??? which stores information about every experiment run. The folder is arranged into sub-folders, one for each day, where in each of those subfolders each experiment is placed in a folder named the exact time of the run. In each experiment folder there is a config file which describes the exact configurations that experiment was run with. These config files are made using hydra. In order to reproduce an experiment, one would have to get the config file from the experiment folder, and use that config file as the config file for a new experiment, as described in question 13. ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- ![my_image](figures/Team17_wandb.PNG)

 We integrated W&B into our project to log various aspects of our model and training. We used W&B to log training error, validation error, test error, and loss. We have set it up such that whenever a new experiment is run, W&B will automatically log the data to our W&B project. ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- For our project we developed 2 images: training, and inference. We have created a Docker File for training the model. 
The TRAINING container is set up to run the hydra configuration. Thus, it trains the model with hyperparameters in config on train data,
    saves the model and evaluates it on test data. Its entrypoint is ENTRYPOINT ["python", "-u", "src/models/train_model.py", "hydra.job.chdir=False"] the image is built on PR from TEST to MASTER, once the image is built the training container can be run with the VERTEX AI with a specific config file that we have set up and can be changed.
    
The INFERENCE container is encapsulating the Fast API. To run the container we are running *docker run -p 80:80 <image>*. This allowed us to easily test the model predictions, by sending HTTP requests.
 
**Dockerfile.train**: https://github.com/capsabogdan/Project-MLOps/blob/dev/Dockerfile.train
      
**Dockerfile.inference**: https://github.com/capsabogdan/Project-MLOps/blob/dev/Dockerfile.inference ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- For debugging, we mainly used printing, as this was a quick way for finding hidden behaviour in the code. We have also used *vscode* debug breakpoints, especially as we had issues with the model returning an empty list of movies.
Finally, we have used experiment logging, creating a W&B report, to better understand the model???s training. ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- We used the following services: Compute Engine, Cloud Storage, Container Registry, VertexAI. Compute Engine is used for creating virtual machines which are highly configurable, e.g. you can configure what hardware you want. Cloud storage is where you can create "buckets" to store data. Container Registry is used for storing containers such as docker images. Vertex AI is used to build and train the model on datasets stored in Cloud Storage  ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- We have used the Google Compute Engine in Various ways. 
Our main VM machine has the following characteristics CPU op-mode(s): 32-bit, 64-bit | CPU(s): 2 | SDD: 150 GB

Services
 * *VertexAI* for dynamically running the trainer container.  
* *Buckets* of Cloud Storage for storing the data and our model checkpoints
* *Cloud Build* for creating triggers spinning the Training container. We also used this service to expose the Inference container in a web service API.

Finally, some of us have used VM instances for developing purposes, connecting their VS session with the remote. ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- ![my_image](figures/Team17_buckets.PNG) ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- ![my_image](figures/Team17_registry.PNG) ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- ![my_image](figures/build.png) ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer: 

--- For the model deployment, we have created a Docker file, creating an image that, containerizes the training, when built. We have pushed this image to the container registry and set up a pipeline, where pull requests from *test* branch to *main* would trigger the entire built, through VertexAI.
To invoke the service , the user needs to call curl -X GET "https://gcp-movie-app-v3-qp7ixpl7fq-ew.a.run.app/predict/100" ---


### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- We did not implement monitoring, but our idea would be to implement Data Drifting monitoring, creating reports with *evidently*, which may help in debugging, exploring and reporting the model???s results
Additionally, we would like to monitor our system, through the use of telemetry. Thus, in our implementation, we would integrate *opentelemetry* with our *fastapi* to extract relevant data and then visualize them through *Signoz*. ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- We ended up using between around 40-60$ with.the most expensive being Cloud Logging,  Cloud Functions, and Cloud Run. ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- ![my_image](figures/Team17_architecture.png)
      
The project structure is based on the cookieCutter template. We are making use of DVC to pull the raw data from the Cloud Buckets, then process the data through the make file and push it to a separate Bucket. Besides, we have set up Github Actions for Continuous Integration and deployed both training and inference as Docker containers in the Cloud Registry. The training container gets built for every pull request from *test* branch to *main*, through the *VertexAI* service, which will run the training, and output the model???s performance.    
Finally, the inference is executed through the fastAPI, which exposes the Docker Inference image in a WebService in the CloudRun Service  - https://gcp-movie-app-v3-qp7ixpl7fq-ew.a.run.app ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- The main struggle were the security and authorization, managing the complex pipeline and the large number of dependencies, especially with regards to the great size of certain packages, such as *torch-geometric* and *torch-geometric*. Some other struggles were the code versioning, merge conflicts, different hardware architectures, reflected in the uniqueness of the docker containers. Another challenge was to set up remote connection from VSCode to the cloud VM instance.  ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- Alexandra Polymenopoulou s212558 was in charge of: Training Deployment, modelling, W&B, Hydra
      
Bogdan Capsa s210172 was in charge of: FASTApi Deployment, GithubActions, DVC
      
Jakob Fahl s184419 was in charge of W&B, MakeData, DVC, modelling
      
Melina Siskou s213158 was in charge of: FASTApi Deployment, MakeData, unit tests 
      
Thomas Spyrou s213161 was in charge of Docker Training Deployment, GithubActions, unit tests. ---


