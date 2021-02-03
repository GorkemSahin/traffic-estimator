import React from 'react';
import { Form, Button, Upload, DatePicker } from 'antd';

const MonthPicker = ({ setFile }) => {

  const handleUpload = (file) => {
    const reader = new FileReader();
    reader.onload = e => {
        setFile(e.target.result);
    };
    reader.readAsText(file);
    return false;
  }

  return (
    <>
      <Form.Item
        labelCol={{ span: 10 }}
        label="Start date"
        name="start"
        rules={[{ required: true }]}
      >
        <DatePicker/>
      </Form.Item>
      <Form.Item
        labelCol={{ span: 10 }}
        label="Prediction date"
        name="test_start"
        rules={[{ required: true }]}
      >
        <DatePicker/>
      </Form.Item>
      <Form.Item
        labelCol={{ span: 10 }}
        label="End date"
        name="end"
        rules={[{ required: true }]}
      >
        <DatePicker/>
      </Form.Item>
      <Form.Item wrapperCol={{ offset: 10, span: 16 }}>
        <Upload beforeUpload={handleUpload}>
          <Button>Select file</Button>
        </Upload>
      </Form.Item>
    </>
  )
}

export default MonthPicker;