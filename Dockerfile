FROM public.ecr.aws/lambda/python:3.8

COPY app/requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

COPY app ${LAMBDA_TASK_ROOT}

ENV AWS_ACCESS_KEY_ID AKIAYZUK3X7VWGS6VAC2
ENV AWS_SECRET_ACCESS_KEY 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
ENV AWS_DEFAULT_REGION ap-northeast-1

CMD [ "neural_style.neural_style.handler" ]
