import { useTranslate } from '@/hooks/common-hooks';
import {
  useDeleteDocument,
  useFetchDocumentInfosByIds,
  useRemoveNextDocument,
  useRunNextDocument,
  useUploadAndParseDocument,
} from '@/hooks/document-hooks';
import kbService from '@/services/knowledge-service';
import { getExtension } from '@/utils/document-util';
import { formatBytes } from '@/utils/file-util';
import {
  CloseCircleOutlined,
  InfoCircleOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import type { GetProp, UploadFile } from 'antd';
import {
  Button,
  Card,
  Flex,
  Input,
  List,
  Popover,
  Space,
  Spin,
  Typography,
  Upload,
  UploadProps,
  notification,
} from 'antd';
import classNames from 'classnames';
import get from 'lodash/get';
import {
  ChangeEvent,
  ChangeEventHandler,
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import FileIcon from '../file-icon';
import SvgIcon from '../svg-icon';
import styles from './index.less';

type FileType = Parameters<GetProp<UploadProps, 'beforeUpload'>>[0];
const { Text } = Typography;

const getFileId = (file: UploadFile) => get(file, 'response.data.0');

const getFileIds = (fileList: UploadFile[]) => {
  const ids = fileList.reduce((pre, cur) => {
    return pre.concat(get(cur, 'response.data', []));
  }, []);

  return ids;
};

const isUploadError = (file: UploadFile) => {
  const retcode = get(file, 'response.retcode');
  return typeof retcode === 'number' && retcode !== 0;
};

const isUploadSuccess = (file: UploadFile) => {
  const retcode = get(file, 'response.retcode');
  return typeof retcode === 'number' && retcode === 0;
};

interface IProps {
  disabled: boolean;
  value: string;
  sendDisabled: boolean;
  sendLoading: boolean;
  onPressEnter(documentIds: string[]): void;
  onInputChange: ChangeEventHandler<HTMLInputElement>;
  conversationId: string;
  uploadMethod?: string;
  isShared?: boolean;
  showUploadIcon?: boolean;
  createConversationBeforeUploadDocument?(message: string): Promise<any>;
  onSelect: ChangeEventHandler<HTMLInputElement>;
}

const getBase64 = (file: FileType): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file as any);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });

const MessageInput = ({
  isShared = false,
  disabled,
  value,
  onPressEnter,
  sendDisabled,
  sendLoading,
  onInputChange,
  conversationId,
  showUploadIcon = true,
  createConversationBeforeUploadDocument,
  uploadMethod = 'upload_and_parse',
  onSelect, // 添加 onSelect 回调属性
}: IProps) => {
  const { t,i18n } = useTranslate('chat');
  const { removeDocument } = useRemoveNextDocument();
  const { deleteDocument } = useDeleteDocument();
  const { data: documentInfos, setDocumentIds } = useFetchDocumentInfosByIds();
  const { uploadAndParseDocument } = useUploadAndParseDocument(uploadMethod);
  const conversationIdRef = useRef(conversationId);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const optionsZh = [
    '自由对话',
    '程序开发',
    '知识库',
    '日志分析',
    'CMDB',
  ]; // 选择框的选项
  const optionsEn = [
    'CHAT',
    'CODE',
    'KNOWLEDGE',
    'LOG',
    'CMDB',
  ]; // 选择框的选项
 

  const [popoverVisible, setPopoverVisible] = useState(false); // 控制 Popover 是否可见
  const [inputValue, setInputValue] = useState(value); // 输入框的值
  const [placeholderValue, setPlaceholderValue] = useState( t('')); // 输入框的值
  console.log(placeholderValue)

  const handlePreview = async (file: UploadFile) => {
    if (!file.url && !file.preview) {
      file.preview = await getBase64(file.originFileObj as FileType);
    }
  };

  // 处理输入框内容改变
  const handleChangeInput = (e: ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);

    // 如果输入框的值以 "@" 开头，显示 Popover
    if (newValue.startsWith('@')) {
      setPopoverVisible(true);
    } else {
      setPopoverVisible(false);
    }
    if (onInputChange) {
      if (newValue.startsWith('@')){
        e.target.value=''
        onInputChange(e); // 调用传递的输入改变处理函数
      }else{
        onInputChange(e); // 调用传递的输入改变处理函数
      }
      
    }
  };

  // 当选择一个选项时
  const handleSelect = (option: ChangeEvent<HTMLInputElement>) => {
    const newValue = inputValue.replace(/@\S*$/, `@${option} `); // 替换 "@xxxx" 为 "@选中的选项"
    // setInputValue(newValue);
    // setPopoverVisible(false);
    if (onSelect) {
      onSelect(option); // 调用 onSelect 回调，将选定的值传递给父组件
    }

    // 使用 Modal 进行提示
    // 显示成功通知
    notification.success({
      message:  t('msg')+` : ${option} `,
      placement: 'topRight', // 可以设置通知的位置，比如 'topRight', 'bottomLeft' 等
    });
    setPopoverVisible(false);

    // 清空输入框的值
    setInputValue('');
    setPlaceholderValue(`@${option}`);
  };

  // const { runDocumentByIds } = useRunNextDocument();
  // const runDocument = useCallback(
  //   (run: number,doc_ids: any) => {
  //     runDocumentByIds({
  //       documentIds: doc_ids,
  //       run,
  //     });
  //   },
  //   [runDocumentByIds],
  // );

  const handleChange: UploadProps['onChange'] = async ({
    //fileList: newFileList,
    file,
  }) => {
    let nextConversationId: string = conversationId;
    console.log('conversationId');
    console.log(conversationId);

    if (createConversationBeforeUploadDocument) {
      const creatingRet = await createConversationBeforeUploadDocument(
        file.name,
      );
      if (creatingRet?.retcode === 0) {
        nextConversationId = creatingRet.data.id;
      }
    }

    setFileList((list) => {
      list.push({
        ...file,
        status: 'uploading',
        originFileObj: file as any,
      });
      return [...list];
    });

    console.log('start upload');

    const ret = await uploadAndParseDocument({
      conversationId: nextConversationId,
      fileList: [file],
    });
    console.log('complete upload');
    setFileList((list) => {
      const nextList = list.filter((x) => x.uid !== file.uid);
      nextList.push({
        ...file,
        originFileObj: file as any,
        response: ret,
        percent: 100,
        status: ret?.retcode === 0 ? 'done' : 'error',
      });
      console.log(nextList)
      return nextList;
    });
    // if (ret?.retcode === 0) {
    //   //console.log('start parse');
    //   const ids = ret.data;
    //   console.log(ids);
    //   //runDocument(1, ids);
    //   setFileList((list) => {
    //     const nextList = list.filter((x) => x.uid !== file.uid);
    //     nextList.push({
    //       ...file,
    //       originFileObj: file as any,
    //       response: ret,
    //       percent: 100,
    //       status: ret?.retcode === 0 ? 'done' : 'error',
    //     });
    //     return nextList;
    //   });
    //   // Start polling document info
    //   startPolling(ids);
    // }
  };

  // const getDocumentInfoById = useCallback(
  //   (id: string) => {
  //     return documentInfos.find((x) => x.id === id);
  //   },
  //   [documentInfos],
  // );

  // const POLLING_INTERVAL = 5000; // Polling every 5 seconds (adjust as necessary)

  // const fetchDocumentInfo = async (ids: string[]) => {
  //   try {
  //     const { data } = await kbService.document_infos({ doc_ids: ids });
  //     if (data.retcode === 0) {
  //       return data.data;
  //     }
  //     return [];
  //   } catch (error) {
  //     console.error('Failed to fetch document info:', error);
  //     return [];
  //   }
  // };

  // Start polling function
  // const startPolling = (ids: string[]) => {
  //   if (intervalRef.current) {
  //     clearInterval(intervalRef.current);
  //   }
  //   intervalRef.current = setInterval(async () => {
  //     console.log('startPolling');
  //     const data = await fetchDocumentInfo(ids);
  //     console.log(data);

  //     if (data.length > 0 && data[0].progress === 1) {
  //       // Stop polling when progress is 1 (complete)
  //       clearInterval(intervalRef.current as NodeJS.Timeout);
  //       intervalRef.current = null;

  //       // Update the file status to 'done'
  //       setFileList((fileList) => {
  //         return fileList.map((file) => {
  //           if (getFileId(file) === ids[0]) {
  //             return {
  //               ...file,
  //               percent: 100,
  //               status: 'done',
  //             };
  //           }
  //           return file;
  //         });
  //       });
  //     }
  //   }, POLLING_INTERVAL);
  // };

  // Cleanup polling on component unmount
  // useEffect(() => {
  //   return () => {
  //     if (intervalRef.current) {
  //       clearInterval(intervalRef.current);
  //     }
  //   };
  // }, []);

  // 同步外部的 value 到内部的 inputValue
  useEffect(() => {
      setInputValue(value);
  }, [value]);
  
  const isUploadingFile = fileList.some((x) => x.status === 'uploading');

  const handlePressEnter = useCallback(async () => {
    if (isUploadingFile) return;
    const ids = getFileIds(fileList.filter((x) => isUploadSuccess(x)));

    onPressEnter(ids);
    setFileList([]);
  }, [fileList, onPressEnter, isUploadingFile]);

  const handleRemove = useCallback(
    async (file: UploadFile) => {
      const ids = get(file, 'response.data', []);
      // Upload Successfully
      if (Array.isArray(ids) && ids.length) {
        if (isShared) {
          await deleteDocument(ids);
        } else {
          await removeDocument(ids[0]);
        }
        setFileList((preList) => {
          return preList.filter((x) => getFileId(x) !== ids[0]);
        });
      } else {
        // Upload failed
        setFileList((preList) => {
          return preList.filter((x) => x.uid !== file.uid);
        });
      }
    },
    [removeDocument, deleteDocument, isShared],
  );

  // const getDocumentInfoById = useCallback(
  //   (id: string) => {
  //     return documentInfos.find((x) => x.id === id);
  //   },
  //   [documentInfos],
  // );

  useEffect(() => {
    const ids = getFileIds(fileList);
    setDocumentIds(ids);
  }, [fileList, setDocumentIds]);

  useEffect(() => {
    if (
      conversationIdRef.current &&
      conversationId !== conversationIdRef.current
    ) {
      setFileList([]);
    }
conversationIdRef.current = conversationId;
  }, [conversationId, setFileList]);

  return (
    <Flex vertical className={styles.messageInputWrapper}>
      <Popover
        content={
          <List
            dataSource={ i18n.language === 'zh' ? optionsZh : optionsEn}
            renderItem={(item: ChangeEvent<HTMLInputElement>) => (
              <List.Item
                onClick={() => handleSelect(item)}
                style={{ cursor: 'pointer' }}
              >
                {item}
              </List.Item>
            )}
          />
        }
        title= {t('selectSkillHolder')}
        trigger="click"
        open={popoverVisible}
        onOpenChange={setPopoverVisible}
        placement="topLeft"
      ></Popover>
      <Input
        size="large"
        placeholder={ placeholderValue =='chat.' ? (i18n.language === 'en' ? 'Call @ IntelliOps at any time, using  skills...' :'随时@小吉, 使用各种能力...' ):placeholderValue }
        value={value}
        disabled={disabled}
        className={classNames({ [styles.inputWrapper]: fileList.length === 0 })}
        suffix={
          <Space>
            {showUploadIcon && (
              <Upload
                // action={uploadUrl}
                // fileList={fileList}
                onPreview={handlePreview}
                onChange={handleChange}
                multiple={false}
                // headers={{ [Authorization]: getAuthorization() }}
                // data={{ conversation_id: conversationId }}
                // method="post"
                onRemove={handleRemove}
                showUploadList={false}
                beforeUpload={(file, fileList) => {
                  console.log('🚀 ~ beforeUpload:', fileList);
                  return false;
                }}
              >
                <Button
                  type={'text'}
                  disabled={disabled}
                  icon={
                    <SvgIcon
                      name="paper-clip"
                      width={18}
                      height={22}
                      disabled={disabled}
                    ></SvgIcon>
                  }
                ></Button>
              </Upload>
            )}
            <Button
              type="primary"
              onClick={handlePressEnter}
              loading={sendLoading}
              disabled={sendDisabled || isUploadingFile}
            >
              {t('send')}
            </Button>
          </Space>
        }
        onPressEnter={handlePressEnter}
        onChange={handleChangeInput}
        onFocus={() => setPopoverVisible(false)} // 当输入框聚焦时隐藏 Popover
      />

      {fileList.length > 0 && (
        <List
          grid={{
            gutter: 16,
            xs: 1,
            sm: 1,
            md: 1,
            lg: 1,
            xl: 2,
            xxl: 4,
          }}
          dataSource={fileList}
          className={styles.listWrapper}
          renderItem={(item) => {
            const id = getFileId(item);
            //const documentInfo = getDocumentInfoById(id);
            const fileExtension = getExtension(item.originFileObj?.name ?? '');
            const fileName = item.originFileObj?.name ?? '';

            return (
              <List.Item>
                <Card className={styles.documentCard}>
                  <Flex gap={10} align="center">
                    {item.status === 'uploading' || !item.response ? (
                      <Spin
                        indicator={
                          <LoadingOutlined style={{ fontSize: 24 }} spin />
                        }
                      />
                    ) : !getFileId(item) ? (
                      <InfoCircleOutlined
                        size={30}
                        // width={30}
                      ></InfoCircleOutlined>
                    ) : (
                      <FileIcon id={id} name={fileName}></FileIcon>
                    )}
                    <Flex vertical style={{ width: '90%' }}>
                      <Text
                        ellipsis={{ tooltip: fileName }}
                        className={styles.nameText}
                      >
                        <b> {fileName}</b>
                      </Text>
                      {isUploadError(item) ? (
                        t('uploadFailed')
                      ) : (
                        <>
                          {item.percent !== 100 ? (
                            t('uploading')
                          ) : !item.response ? (
                            t('parsing')
                          ) : (
                            <Space>
                              <span>{fileExtension?.toUpperCase()},</span>
                              <span>
                                {formatBytes(
                                  item.originFileObj?.size ?? 0,
                                )}
                              </span>
                            </Space>
                          )}
                        </>
                      )}
                    </Flex>
                  </Flex>

                  {item.status !== 'uploading' && (
                    <span className={styles.deleteIcon}>
                      <CloseCircleOutlined onClick={() => handleRemove(item)} />
                    </span>
                  )}
                </Card>
              </List.Item>
            );
          }}
        />
      )}
    </Flex>
  );
};

export default memo(MessageInput);
