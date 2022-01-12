FROM public.ecr.aws/lambda/python:3.8

COPY app/requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

COPY app ${LAMBDA_TASK_ROOT}

CMD [ "neural_style.neural_style.handler" ]
